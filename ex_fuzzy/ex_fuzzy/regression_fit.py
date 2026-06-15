"""
Evolutionary Optimization for Fuzzy Rule Base Learning (Regression)

This module implements genetic algorithm-based optimization for learning fuzzy
rule bases for regression problems. It mirrors the classification machinery in
``evolutionary_fit.py`` (``FitRuleBase`` + ``BaseFuzzyRulesClassifier``) but uses
Mamdani-style scalar consequents and weighted-average (height) defuzzification.

Main Components:
    - RuleBaseT1Regression: Scalar-consequent rule base with weighted-average inference
    - FitRuleBaseRegression: Core optimization problem class for genetic algorithms
    - BaseFuzzyRulesRegressor: Scikit-learn compatible regressor (fit/predict/score)

The module supports automatic learning of:
    - Rule antecedents (which variables and linguistic terms to use)
    - Rule consequents (scalar output values, encoded into the target range)
    - Rule structure (number of rules, antecedent count)

Key Features:
    - Vectorized fitness evaluation: the linguistic variables are fixed during the
      search, so antecedent memberships are precomputed once and every candidate is
      scored with pure-numpy gather + product (no per-evaluation rule-base objects
      and no Python rule loop), exactly like the classifier's fast path.
    - Cross-validation or full-train R2 fitness.
    - Weighted-average (height) defuzzification with mean fallback and range clamping.
    - Support for Type-1 fuzzy systems.
"""
import numpy as np
import pandas as pd

from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.model_selection import KFold

from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation

try:
    from . import fuzzy_sets as fs
    from . import rules
    from . import utils
except ImportError:
    import fuzzy_sets as fs
    import rules
    import utils



class RuleBaseT1Regression():
    '''
    Type-1 fuzzy rule base for regression. Each rule carries a scalar consequent.

    The crisp output is the firing-weighted average of the rule consequents
    (Mamdani height defuzzification), with a fallback to the training target mean
    for samples that fire no rule, and a final clamp to the training target range.
    '''

    def __init__(self, antecedents: list[fs.fuzzyVariable], rule_list: list[rules.RuleSimple], scalar_consequents: np.array,
                 y_min: float = None, y_max: float = None, y_mean: float = None, clamp_margin: float = 0.05, tnorm=np.prod) -> None:
        '''
        Creates a rule base with the given antecedents, rules and scalar consequents.

        :param antecedents: list of fuzzyVariable used as antecedents (the linguistic variables).
        :param rule_list: list of RuleSimple. Each rule's antecedent indices point to a linguistic term per variable (-1 means "don't care").
        :param scalar_consequents: array with the scalar output value for each rule.
        :param y_min: minimum target value seen in training. Used for the output clamp.
        :param y_max: maximum target value seen in training. Used for the output clamp.
        :param y_mean: mean target value seen in training. Used as the no-fire fallback.
        :param clamp_margin: fraction of the target range allowed beyond [y_min, y_max] before clamping.
        :param tnorm: t-norm used to aggregate antecedent memberships into a rule firing strength.
        '''
        self.antecedents = antecedents
        self.tnorm = tnorm
        self.y_min, self.y_max, self.y_mean = y_min, y_max, y_mean
        self.clamp_margin = clamp_margin
        # Keep-max dedup of rules sharing the same antecedents (identical behaviour
        # to decoding the best individual found by the genetic algorithm).
        self.rules, self.scalar_consequents = self._dedup(rule_list, np.asarray(scalar_consequents, dtype=float))
        for ix, rule in enumerate(self.rules):
            rule.consequent = self.scalar_consequents[ix]


    def _dedup(self, rule_list: list[rules.RuleSimple], cons: np.array) -> tuple:
        '''
        Removes duplicated rules (same antecedents), keeping the largest consequent.

        :param rule_list: list of RuleSimple to filter.
        :param cons: array with the scalar consequent of each rule.
        :return: tuple (unique rule list, array of their scalar consequents).
        '''
        unique = {}
        for ix, rule in enumerate(rule_list):
            key = tuple(rule.antecedents)
            if key not in unique or cons[ix] > unique[key][1]:
                unique[key] = (ix, cons[ix])
        if not unique:
            return [], np.array([])
        kept = list(unique.values())
        return [rule_list[ix] for ix, _ in kept], np.array([c for _, c in kept])


    def get_rules(self) -> list[rules.RuleSimple]:
        '''
        Returns the list of rules in the rule base.
        '''
        return self.rules


    def compute_antecedents_memberships(self, x: np.array) -> list:
        '''
        Computes, for each antecedent variable, the membership of every sample to each of its linguistic terms.

        :param x: array with the values of the inputs. Shape (samples, features).
        :return: list with one entry per antecedent variable; each entry has shape (n_terms, samples).
        '''
        if hasattr(x, 'values'):
            x = x.values
        return [ant.compute_memberships(x[:, ix]) for ix, ant in enumerate(self.antecedents)]


    def compute_rule_antecedent_memberships(self, x: np.array, cached: list = None) -> np.array:
        '''
        Computes the firing strength of every rule for every sample.

        :param x: array with the values of the inputs. Shape (samples, features).
        :param cached: optional precomputed antecedent memberships (see compute_antecedents_memberships).
        :return: array of shape (samples, rules) with the firing strength of each rule.
        '''
        cached = cached or self.compute_antecedents_memberships(x)
        res = np.zeros((x.shape[0], len(self.rules)))
        for jx, rule in enumerate(self.rules):
            membership = np.ones((x.shape[0], len(rule.antecedents)))
            n_active = 0
            for ix, vl in enumerate(rule.antecedents):
                if vl >= 0:
                    membership[:, ix] = list(cached[ix])[vl]
                    n_active += 1
            # A rule with no active antecedent ("all don't care") does not fire.
            if n_active == 0:
                membership[:] = 0.0
            res[:, jx] = self.tnorm(membership, axis=1)
        return res


    def inference(self, x: np.array, cached: list = None) -> np.array:
        '''
        Computes the crisp regression output for every sample.

        :param x: array with the values of the inputs. Shape (samples, features).
        :param cached: optional precomputed antecedent memberships.
        :return: vector of size (samples,) with the predicted output.
        '''
        if len(self.rules) == 0:
            return np.full(x.shape[0], self.y_mean if self.y_mean is not None else 0.0)

        firing = self.compute_rule_antecedent_memberships(x, cached)
        den = np.sum(firing, axis=1)
        res = np.full(x.shape[0], self.y_mean if self.y_mean is not None else 0.0)

        # Weighted average of the scalar consequents; samples that fire no rule
        # keep the training mean fallback.
        mask = den > 1e-10
        res[mask] = (firing[mask] @ self.scalar_consequents) / den[mask]

        if self.y_min is not None and self.y_max is not None:
            rng = self.y_max - self.y_min
            res = np.clip(res, self.y_min - self.clamp_margin * rng, self.y_max + self.clamp_margin * rng)
        return res


    def predict(self, x: np.array) -> np.array:
        '''
        Alias of inference. Returns the predicted output for every sample.

        :param x: array with the values of the inputs. Shape (samples, features).
        :return: vector of size (samples,) with the predicted output.
        '''
        return self.inference(x)


    def print_rules(self, return_rules: bool = False):
        '''
        Prints (or returns) the rule base in a human-readable IF-THEN form.

        :param return_rules: if True, the rules are returned as a string instead of printed.
        :return: the rules as a string if return_rules is True, otherwise None.
        '''
        out = ''
        for rule in self.rules:
            parts = [f"{ant.name} IS {ant.linguistic_variable_names()[rule[j]]}"
                     for j, ant in enumerate(self.antecedents) if rule[j] >= 0]
            out += 'IF ' + ' AND '.join(parts) + f' THEN output = {rule.consequent:.4f}\n'
        return out if return_rules else print(out)


    def __len__(self) -> int:
        return len(self.rules)

    def __getitem__(self, ix: int) -> rules.RuleSimple:
        return self.rules[ix]

    def __iter__(self):
        return iter(self.rules)



class FitRuleBaseRegression(Problem):
    '''
    Pymoo problem that optimizes a regression rule base over a fixed set of
    linguistic variables, using the vectorized fast-scoring scheme of the classifier.

    Chromosome layout (all integer-valued):
        [ feature index    ]  nRules * nAnts   in [0, n_features - 1]
        [ term index       ]  nRules * nAnts   in [-1, max_terms - 1]   (-1 = don't care)
        [ scalar consequent]  nRules            in [0, 100]   (encoded into the target range)
    '''

    def __init__(self, X: np.array, y: np.array, nRules: int, nAnts: int, linguistic_variables: list[fs.fuzzyVariable],
                 tolerance: float = 0.001, use_cv_fitness: bool = True, cv_folds: int = 3,
                 y_min: float = None, y_max: float = None, y_mean: float = None) -> None:
        '''
        Inits the optimization problem with the data and the (fixed) linguistic variables.

        :param X: numpy array or dataframe samples x features.
        :param y: target vector. float array samples (x 1).
        :param nRules: number of rules to optimize.
        :param nAnts: max number of antecedents per rule.
        :param linguistic_variables: list of fuzzyVariable used as antecedents. They are kept fixed during the search.
        :param tolerance: tolerance for the rule scoring/pruning hooks.
        :param use_cv_fitness: if True, the fitness is the mean k-fold R2; otherwise the full-train R2.
        :param cv_folds: number of folds for the cross-validation fitness.
        :param y_min: minimum target value. If None (default) it is computed empirically.
        :param y_max: maximum target value. If None (default) it is computed empirically.
        :param y_mean: mean target value. If None (default) it is computed empirically.
        '''
        try:
            self.var_names = list(X.columns)
            self.X = X.values
        except AttributeError:
            self.X = np.asarray(X)
            self.var_names = [str(ix) for ix in range(X.shape[1])]

        # The fast scoring path precomputes a dense membership tensor, which is only
        # defined for Type-1 fuzzy sets.
        if linguistic_variables[0].fuzzy_type() != fs.FUZZY_SETS.t1:
            raise ValueError("FitRuleBaseRegression only supports Type-1 (t1) fuzzy sets.")

        self.y = np.asarray(y, dtype=float)
        self.lvs = linguistic_variables
        self.n_lv = [len(lv.linguistic_variable_names()) for lv in self.lvs]
        self.nRules, self.nAnts = nRules, nAnts
        self.tolerance = tolerance
        self.use_cv_fitness, self.cv_folds = use_cv_fitness, cv_folds

        self.y_min = float(np.min(self.y)) if y_min is None else y_min
        self.y_max = float(np.max(self.y)) if y_max is None else y_max
        self.y_mean = float(np.mean(self.y)) if y_mean is None else y_mean
        self.y_range = (self.y_max - self.y_min) or 1.0

        # Build the integer bounds for each gene segment.
        bounds = []
        for _ in range(nRules):
            for _ in range(nAnts):
                bounds.append([0, self.X.shape[1] - 1])          # feature index
        for _ in range(nRules):
            for _ in range(nAnts):
                bounds.append([-1, max(self.n_lv) - 1])          # term index (-1 = don't care)
        for _ in range(nRules):
            bounds.append([0, 100])                              # scalar consequent (encoded)
        bounds = np.array(bounds)
        super().__init__(n_var=len(bounds), n_obj=1, xl=bounds[:, 0], xu=bounds[:, 1])

        self._precompute()


    def _precompute(self) -> None:
        '''
        Precomputes the quantities reused by every fitness evaluation: the dense
        membership tensor, the gather index helper, the output clamp and the CV splits.

        :return: None. Results are stored as attributes.
        '''
        n_samples, n_features = self.X.shape
        self._n_lv_per_feat = np.array(self.n_lv, dtype=int)
        max_lvars = int(self._n_lv_per_feat.max())
        self._max_lvars = max_lvars
        # Last channel is a constant 1.0 used as the "don't care" antecedent so the
        # per-feature product naturally ignores inactive variables.
        self._dont_care = max_lvars

        membership_array = np.zeros((n_samples, n_features, max_lvars + 1))
        membership_array[:, :, max_lvars] = 1.0
        for f in range(n_features):
            mems = self.lvs[f].compute_memberships(self.X[:, f])  # (n_lvars, n_samples)
            membership_array[:, f, :mems.shape[0]] = mems.T
        self._membership_array = membership_array
        self._feat_index = np.broadcast_to(np.arange(n_features)[None, :], (self.nRules, n_features))

        self._clamp_lo = self.y_min - 0.05 * self.y_range
        self._clamp_hi = self.y_max + 0.05 * self.y_range

        if self.use_cv_fitness:
            kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            self._cv_splits = []
            for _, val_idx in kf.split(self.X):
                y_val = self.y[val_idx]
                self._cv_splits.append((val_idx, y_val, np.sum((y_val - np.mean(y_val)) ** 2)))


    def _rule_term_matrix(self, xi: np.array) -> np.array:
        '''
        Decodes a chromosome into a (nRules, n_features) matrix of selected term indices.

        Each entry is the linguistic term chosen for that (rule, feature) pair, or
        the "don't care" channel. Repeated features within a rule follow a
        last-slot-wins rule, matching the object-based construction.

        :param xi: integer chromosome (single individual).
        :return: integer matrix of shape (nRules, n_features).
        '''
        nR, nA, nF = self.nRules, self.nAnts, self.X.shape[1]
        chosen_ants = np.clip(xi[:nA * nR].reshape(nR, nA), 0, nF - 1)
        term_idx = xi[nA * nR:2 * nA * nR].reshape(nR, nA)
        max_term = self._n_lv_per_feat[chosen_ants] - 1
        valid = (term_idx >= 0) & (term_idx <= max_term)

        rule_term = np.full((nR, nF), self._dont_care, dtype=int)
        rows = np.arange(nR)
        for a in range(nA):  # nAnts is tiny; last-slot-wins on repeated features
            m = valid[:, a]
            rule_term[rows[m], chosen_ants[m, a]] = term_idx[m, a]
        return rule_term


    def _fast_predict(self, xi: np.array) -> np.array:
        '''
        Vectorized weighted-average prediction for a single chromosome.

        Mirrors RuleBaseT1Regression.inference (tnorm = product) but without building
        any rule-base object: it gathers the precomputed memberships and reduces them
        with a product over features, then defuzzifies by weighted average.

        :param xi: integer chromosome (single individual).
        :return: predictions over all training samples.
        '''
        nR = self.nRules
        rule_term = self._rule_term_matrix(xi)
        n_active = np.sum(rule_term != self._dont_care, axis=1)

        # gathered[s, r, f] = membership_array[s, f, rule_term[r, f]] -> product over f.
        gathered = self._membership_array[:, self._feat_index, rule_term]  # (S, nR, nF)
        firing = np.prod(gathered, axis=2)                                 # (S, nR)
        firing[:, n_active == 0] = 0.0

        cons_enc = xi[2 * self.nAnts * nR:2 * self.nAnts * nR + nR]
        cons = self.y_min + (cons_enc / 100.0) * self.y_range

        num = firing @ cons
        den = np.sum(firing, axis=1)
        preds = np.full(self.X.shape[0], self.y_mean, dtype=float)
        mask = den > 1e-10
        preds[mask] = num[mask] / den[mask]
        return np.clip(preds, self._clamp_lo, self._clamp_hi)


    def _r2(self, y_true: np.array, preds: np.array, ss_tot: float) -> float:
        '''
        Coefficient of determination given a precomputed total sum of squares.

        :param y_true: true target values.
        :param preds: predicted values.
        :param ss_tot: total sum of squares of y_true.
        :return: the R2 score (0.0 when ss_tot is not positive).
        '''
        ss_res = np.sum((y_true - preds) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0.0


    def _cv_fitness(self, preds: np.array) -> float:
        '''
        Mean k-fold R2 of the given predictions over the precomputed CV splits.

        :param preds: predictions for all training samples.
        :return: mean R2 across folds.
        '''
        return float(np.mean([self._r2(y_val, preds[val_idx], ss_tot)
                              for val_idx, y_val, ss_tot in self._cv_splits]))


    def _train_fitness(self, preds: np.array) -> float:
        '''
        Full-train R2 of the given predictions.

        :param preds: predictions for all training samples.
        :return: the R2 score on the whole training set.
        '''
        return self._r2(self.y, preds, np.sum((self.y - np.mean(self.y)) ** 2))


    def _evaluate(self, x: np.array, out: dict, *args, **kwargs) -> None:
        '''
        Pymoo fitness callback. Scores every individual in the population.

        :param x: population array of shape (pop_size, n_var).
        :param out: dict whose "F" field receives the (negated) fitness, since pymoo minimizes.
        :return: None. The result is written into out["F"].
        '''
        fitness = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            xi = np.round(x[i]).astype(int)
            preds = self._fast_predict(xi)
            fitness[i] = self._cv_fitness(preds) if self.use_cv_fitness else self._train_fitness(preds)
        out["F"] = -fitness


    def _encode_cons(self, val: float) -> int:
        '''
        Encodes a scalar consequent into the [0, 100] gene range.

        :param val: scalar consequent value.
        :return: integer encoding in [0, 100].
        '''
        return int(np.clip((val - self.y_min) / self.y_range * 100, 0, 100))


    def _decode_cons(self, encoded: int) -> float:
        '''
        Decodes a [0, 100] gene value back into a scalar consequent.

        :param encoded: integer encoding in [0, 100].
        :return: the scalar consequent value.
        '''
        return self.y_min + (encoded / 100.0) * self.y_range


    def _construct_ruleBase(self, x: np.array) -> RuleBaseT1Regression:
        '''
        Decodes a chromosome into a concrete RuleBaseT1Regression object.

        Called once, on the best individual, after the genetic search. It uses the
        same antecedent/consequent decoding as the fast scoring path, so the deployed
        model reproduces what was scored.

        :param x: integer chromosome (single individual).
        :return: the decoded rule base.
        '''
        x = np.round(x).astype(int)
        nR, nA = self.nRules, self.nAnts
        rule_list, consequents = [], []
        for r in range(nR):
            ant = [-1] * len(self.lvs)
            n_active = 0
            for a in range(nA):
                feat_idx = x[r * nA + a]
                term_idx = x[nR * nA + r * nA + a]
                if term_idx >= 0 and feat_idx < len(self.lvs):
                    if term_idx < self.n_lv[feat_idx]:
                        ant[feat_idx] = term_idx  # last-slot-wins on repeated features
                        n_active += 1
            if n_active > 0:
                rule_list.append(rules.RuleSimple(ant, 0))
                consequents.append(self._decode_cons(x[2 * nR * nA + r]))
        return RuleBaseT1Regression(self.lvs, rule_list,
                                    np.array(consequents) if consequents else np.array([]),
                                    y_min=self.y_min, y_max=self.y_max, y_mean=self.y_mean)



class BaseFuzzyRulesRegressor(RegressorMixin, BaseEstimator):
    '''
    Class that is used as a regressor for a fuzzy rule based system. Supports
    precomputed linguistic variables or automatic partitioning of the inputs.
    '''

    def __init__(self, nRules: int = 30, nAnts: int = 4, n_linguistic_variables: int = 3,
                 fuzzy_type: fs.FUZZY_SETS = fs.FUZZY_SETS.t1, linguistic_variables: list[fs.fuzzyVariable] = None,
                 tolerance: float = 0.001, use_cv_fitness: bool = True, cv_folds: int = 3, verbose: bool = False) -> None:
        '''
        Inits the regressor with the corresponding parameters.

        :param nRules: number of rules to optimize.
        :param nAnts: max number of antecedents per rule.
        :param n_linguistic_variables: number of fuzzy sets per variable when partitions are auto-built.
        :param fuzzy_type: FUZZY_SETS enum type. Only Type-1 (t1) is supported.
        :param linguistic_variables: list of fuzzyVariable. If None (default) partitions are built from the data with utils.construct_partitions.
        :param tolerance: tolerance for the rule scoring/pruning hooks.
        :param use_cv_fitness: if True, the fitness is the mean k-fold R2; otherwise the full-train R2.
        :param cv_folds: number of folds for the cross-validation fitness.
        :param verbose: if True, prints the progress of the optimization.
        '''
        self.nRules = nRules
        self.nAnts = nAnts
        self.n_linguistic_variables = n_linguistic_variables
        self.fuzzy_type = fuzzy_type
        self.linguistic_variables = linguistic_variables
        self.tolerance = tolerance
        self.use_cv_fitness = use_cv_fitness
        self.cv_folds = cv_folds
        self.verbose = verbose

        self.rule_base = None
        self.lvs = None
        self.performance = None
        self._y_min = self._y_max = self._y_mean = None
        self.feature_names = None


    def fit(self, X: np.array, y: np.array, n_gen: int = 50, pop_size: int = 50, random_state: int = 42):
        '''
        Fits a fuzzy rule based regressor using a genetic algorithm to the given data.

        :param X: numpy array or dataframe samples x features.
        :param y: target vector. float array samples (x 1).
        :param n_gen: integer. Number of generations to run the genetic algorithm.
        :param pop_size: integer. Population size for each generation.
        :param random_state: integer. Random seed for the optimization process.
        :return: self, the fitted regressor.
        '''
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            Xv = X.values.astype(float)
        else:
            Xv = np.asarray(X, dtype=float)
            self.feature_names = [f'x{i}' for i in range(Xv.shape[1])]
        y = np.asarray(y, dtype=float)
        self._y_min, self._y_max, self._y_mean = float(y.min()), float(y.max()), float(y.mean())

        # Either use precomputed linguistic variables or build standard partitions.
        if self.linguistic_variables is not None:
            self.lvs = self.linguistic_variables
        else:
            self.lvs = utils.construct_partitions(
                Xv, fz_type_studied=self.fuzzy_type, n_partitions=self.n_linguistic_variables)

        if self.nAnts > len(self.lvs):
            self.nAnts = len(self.lvs)
            if self.verbose:
                print(f'Warning: nAnts higher than the number of variables. Setting nAnts to {len(self.lvs)}.')

        problem = FitRuleBaseRegression(
            Xv, y, self.nRules, self.nAnts, self.lvs,
            tolerance=self.tolerance, use_cv_fitness=self.use_cv_fitness, cv_folds=self.cv_folds,
            y_min=self._y_min, y_max=self._y_max, y_mean=self._y_mean)

        # Standard pymoo GA with default (random) sampling; RoundingRepair keeps the
        # integer-valued genes valid after crossover and mutation.
        algo = GA(pop_size=pop_size,
                  crossover=SBX(prob=0.9, eta=3.0, repair=RoundingRepair()),
                  mutation=PolynomialMutation(eta=7.0, repair=RoundingRepair()),
                  eliminate_duplicates=False)

        res = minimize(problem, algo, ('n_gen', n_gen), seed=random_state,
                       save_history=False, verbose=self.verbose)

        F = res.pop.get('F')
        best = res.pop.get('X')[int(np.argmin(F))]
        self.performance = float(-np.min(F))

        # Decode the best individual once into a concrete rule base.
        self.rule_base = problem._construct_ruleBase(best)
        self.lvs = self.rule_base.antecedents
        if self.verbose:
            print(f"Final: {len(self.rule_base)} rules, fitness={self.performance:.4f}")
        return self


    def predict(self, X: np.array) -> np.array:
        '''
        Returns the predicted output for each sample.

        :param X: numpy array or dataframe samples x features.
        :return: vector of size (samples,) with the predicted output.
        '''
        if self.rule_base is None:
            raise ValueError("Model not fitted")
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.rule_base.predict(np.asarray(X, dtype=float))


    def score(self, X: np.array, y: np.array) -> float:
        '''
        Returns the coefficient of determination (R2) of the prediction.

        :param X: numpy array or dataframe samples x features.
        :param y: true target values.
        :return: the R2 score.
        '''
        preds, y = self.predict(X), np.asarray(y, dtype=float)
        ss_res, ss_tot = np.sum((y - preds) ** 2), np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / (ss_tot + 1e-10) if ss_tot > 0 else 0.0


    def print_rules(self, return_rules: bool = False):
        '''
        Prints (or returns) the rules of the fitted model in IF-THEN form.

        :param return_rules: if True, the rules are returned as a string instead of printed.
        :return: the rules as a string if return_rules is True, otherwise None.
        '''
        if self.rule_base is None:
            return "No rules (model not fitted)"
        return self.rule_base.print_rules(return_rules)


    def get_rulebase(self) -> RuleBaseT1Regression:
        '''
        Returns the fitted RuleBaseT1Regression object.
        '''
        return self.rule_base
