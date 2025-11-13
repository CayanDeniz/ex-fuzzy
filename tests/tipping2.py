import numpy as np
from ex_fuzzy.fuzzy_sets import FS, fuzzyVariable
from ex_fuzzy.rules import RuleSimple, RuleBaseT1

def create_fuzzy_variables():

    food_rancid = FS("Rancid", [0, 0, 1, 5], [0, 9])
    food_delicious = FS("Delicious", [4, 8, 9, 9], [0, 9])
    food_var = fuzzyVariable("Food", [food_rancid, food_delicious], "quality units")

    service_poor = FS("Poor", [0, 0, 1, 3], [0, 9])
    service_good = FS("Good", [1, 3, 5, 7], [0, 9])
    service_excellent = FS("Excellent", [5, 7, 9, 9], [0, 9])
    service_var = fuzzyVariable("Service", [service_poor, service_good, service_excellent], "quality units")

    tip_cheap = FS("Cheap", [0, 6, 6, 12], [0, 30])
    tip_average = FS("Average", [10, 15, 15, 20], [0, 30])
    tip_generous = FS("Generous", [18, 24, 24, 30], [0, 30])
    tip_var = fuzzyVariable("Tip", [tip_cheap, tip_average, tip_generous], "dollars")
    
    return food_var, service_var, tip_var


def create_rules():

    rules = []
    
    rules.append(RuleSimple([0, 0], consequent=0))
    rules.append(RuleSimple([0, 1], consequent=0))
    rules.append(RuleSimple([0, 2], consequent=1))
    rules.append(RuleSimple([1, 0], consequent=1))
    rules.append(RuleSimple([1, 1], consequent=1))
    rules.append(RuleSimple([1, 2], consequent=2))
    
    return rules


def create_fuzzy_system():

    food_var, service_var, tip_var = create_fuzzy_variables()
    
    rules = create_rules()
    
    # using product t-norm from exfuzzy
    rule_base = RuleBaseT1(
        antecedents=[food_var, service_var],
        rules=rules,
        consequent=tip_var,
        tnorm=np.prod
    )
    
    return rule_base

rule_base = create_fuzzy_system()

test_cases = [(4.0, 8.5), (9.0, 6.0), (4.0, 2.5)]

print("Results:")

for i, (food, service) in enumerate(test_cases, 1):
    x = np.array([[food, service]])
    tip = rule_base.inference(x)[0]
    print(f"1.{'abc'[i-1]}. {tip}")