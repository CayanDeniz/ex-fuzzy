import numpy as np
import matplotlib.pyplot as plt

def trapezium_rule(x, param):
    a = param[0]
    b = param[1]
    c = param[2]
    d = param[3]
    
    # Coordindates x is a point and params are x values. In order to find the gradient we substract it fom boundary point then divide by interval of boundaries. 
    # m = (x2 - x1) / (y2 - y1)
    
    if x <= a:
        return 0
    elif a < x <= b:
        result = (x - a) / (b - a)
        return result
    elif b < x <= c:
        return 1.0
    elif c < x <= d:
        result = (d - x) / (d - c)    
        return result 
    else:
        return 0.0
    
class Fuzzify:
    def __init__(self, name, mf):
        self.name = name
        self.mf = mf
    
    def fuzzify(self, value):
        result = {}
        for i, j in self.mf.items():
            result[i] = trapezium_rule(value, j)
            
        return result

food_taste = Fuzzify('food', {'rancid': [0, 0, 1, 5], 'delicious': [4, 8, 9, 9]})
food_service = Fuzzify('service', {'poor': [0, 0, 1, 3], 'good': [1, 3, 5, 7], 'excellent': [5, 7, 9, 9]})
tip_rate = Fuzzify('tip', {'cheap': [0, 6, 6, 12], 'average': [10, 15, 15, 20], 'generous': [18, 24, 24, 30]})
        
def rulebase(taste, service):
    rule1 = min(taste['rancid'], service['poor'])
    rule2 = min(taste['rancid'], service['good'])
    rule3 = min(taste['rancid'], service['excellent'])
    rule4 = min(taste['delicious'], service['poor'])
    rule5 = min(taste['delicious'], service['good'])
    rule6 = min(taste['delicious'], service['excellent'])
    
    aggregation = {'cheap': max(rule1,rule2), 'average': max(rule3,rule4,rule5), 'generous': rule6}
    
    return aggregation

def defuzzify(tip_agg, tip_members):    
    
    tip_range = np.linspace(0, 30, 1000)
    output = np.zeros_like(tip_range)
    
    for i, j in enumerate(tip_range):
        membership = []
        for k, z in tip_members.items():
            base_area = trapezium_rule(j, z)
            
            clipped_area = min(base_area, tip_agg[k])
            membership.append(clipped_area)
        output[i] = max(membership)             
    
    if np.sum(output) == 0:
        return 0
    centroid = np.sum(tip_range * output) / np.sum(output)
    return centroid

def inference(food_value, service_value):
    
    food_fuzzy = food_taste.fuzzify(food_value)
    service_fuzzy = food_service.fuzzify(service_value)
    
    tip_activation = rulebase(food_fuzzy, service_fuzzy)
    crisp_output = defuzzify(tip_activation, tip_rate.mf)
    
    return crisp_output

tip_a = inference(4, 8.5)
tip_b = inference(9, 6)
tip_c = inference(4, 2.5)

print(f"""
    Results:
    
    1.a. {tip_a}
    1.b. {tip_b}
    1.c. {tip_c}   
    
      """)

# *If the food is rancid and the service is poor, then the tip is cheap
# *If the food is rancid and the service is good, then the tip is cheap
# *If the food is rancid and the service is excellent, then the tip is average
# *If the food is delicious and the service is poor, then the tip is average
# *If the food is delicious and the service is good, then the tip is average
# *If the food is delicious and the service is excellent, then the tip is generous

 
# 1a-Food = 4.0, Service = 8.5,
 
 
# 1b- food : 9.0, service : 6.0
 
 
# 1c- food : 4.0, service : 2.5