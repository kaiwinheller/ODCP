# This file contains a function that uses gurobi to solve a ODCP with the following characteristics
# Shape : 2-Dimensional
# depot : in the center
# distance metric : manhattan
# OD Locations : discrete
# OD arrivals : simultanious
# compensation : static
# OD arrivals : potentially many
# customers : potentially many

import numpy as np
import random
from itertools import chain, combinations, product
from scipy.spatial import distance_matrix
import gurobipy as gp

# Helper Function
def binary_tuples(n):
    """
    Generate a set of all possible combinations of binary tuples of size n.
    
    Parameters:
    - n: Integer, size of the tuples.
    
    Returns:
    - A Dictionary (Keys are string versions of the combination and value is an array representation)
    """
    # Use itertools.product to generate all combinations of 0 and 1 of length n
    combinations = product([False, True], repeat=n)
    tuple_list = list(combinations)
    powerset_O = {"".join(map(str, map(int, O_tilde))): np.array(O_tilde) for O_tilde in tuple_list}
    return powerset_O

def optimize_instance_2D(detour_matrix, cost_of_dedicated_delivery, individual_arrival_probabilities, fixed_negative_utility = 0):
    '''
    This function takes several arguments and puts out the optimal solution value (costs) and the optimal compensations to the problem ODCP with the following characteristics
    # Shape : 2-Dimensional
    # depot : in the center
    # distance metric : manhattan
    # OD Locations : discrete
    # OD arrivals : simultanious
    # compensation : static
    # OD arrivals : potentially many
    # customers : potentially many

    Instance Parameters
    ______________________
    detour_matrix: 
    - type: np.array() with shape (O, C) where O is the number of ODs and C is the number of customers
    - description: The detour_matrix describes the detour it takes for OD o to reach customer c and then go to his destination from the depot instead of directly going there (detour = -d[oc] + d[0c] + d[co])

    cost_of_dedicated_delivery:
    - type: numpy array with size C
    - description: The cost_of_dedicated_delivery resembles the costs associated with the delivery of customer c by a dedicated driver that incur when no OD delivers to the customer c

    individual_arrival_probabilities:
    - type: numpy array with size O
    - description: The individual_arrival_probabilities represent the probability that OD o arrives within the predefined timespan

    fixed_negative_utility:
    - type: numeric
    - description: This is an unavoidable negative utility that includes the perceived costs of the OD for e.g. packing the goods, searching a parking lot etc.

    Output
    ______________________
    objective_value:
    - type: float
    - description: The total expected costs when applying the optimal compensation policy

    optimal_compensations:
    - type: dictionary (customer index: compensation)
    '''
    # Parametrers:
    # Number of ODs and Customers
    O, C = np.shape(detour_matrix)
    # All subsets of the set of ODs
    powerset_O = binary_tuples(O)
    powerset_O_w_0 = powerset_O.copy()
    powerset_O_w_0.pop('0'*O)
    # Probability that at least one OD in the set of ODs comes
    acc_probability = {O_tilde: 1 - np.product(([1 - individual_arrival_probabilities[powerset_O[O_tilde]]])) for O_tilde in powerset_O}
    # Cardinality of the subset of all ODs
    cardinality = {O_tilde: np.sum(powerset_O[O_tilde]) for O_tilde in powerset_O}
    # Sufficiently large number (must be higher than the maximum number of ODs and the maximum utility an OD can achieve)
    M = 1000

    # Create the Model
    model = gp.Model("ODCP")

    # Create the Variables
    x = model.addVars(O, C, vtype='B', name='indicator')
    y = model.addVars(powerset_O, C, vtype='B', name='indicator_2') # Make it a binary Tuple
    r = model.addVars(C, lb=0, vtype='C', name='compensation')
    z = model.addVars(C, lb=0, vtype='C', name='maximum_dummy')
    # Terms that directly depend on variables
    u = np.array([[- fixed_negative_utility - detour_matrix[o,c] + r[c] for c in range(C)] for o in range(O)])

    # Create objective function
    model.setObjective(gp.quicksum(gp.quicksum(acc_probability[O_tilde] * y[O_tilde, c] * (cost_of_dedicated_delivery[c] - r[c]) for O_tilde in powerset_O) for c in range(C)), gp.GRB.MAXIMIZE)

    # Create the constraints
    model.addConstrs((z[c] <= u[o,c] + M*(1-x[o,c]) for c in range(C) for o in range(O)), name='C1')
    model.addConstrs((gp.quicksum(x[o,c] for c in range(C)) <= 1 for o in range(O)), name='C2')
    model.addConstrs((gp.quicksum(x[o,c] for o in [index for index, char in enumerate(O_tilde) if char == '1'])/cardinality[O_tilde] + M*(1-y[O_tilde, c]) >= 1 for O_tilde in powerset_O_w_0 for c in range(C)), name='C3')
    model.addConstrs((gp.quicksum(y[O_tilde,c] for O_tilde in powerset_O) == 1 for c in range(C)), name='C4')

    # Optimize
    model.optimize()

    # Output Parameters
    if model.status == gp.GRB.OPTIMAL:
        objective_value = model.ObjVal
        print(f"Objective Value: {objective_value}")

        variable_values = [model.getVarByName(f'compensation[{c}]').X for c in range(C)]
        for i, comp in enumerate(variable_values):
            print(f'Compensation for customer {i} = {comp}')

        assignments = [(o, c) for o in range(O) for c in range(C) if model.getVarByName(f'indicator[{o},{c}]').X == 1]
        for a in assignments:
            print(f'OD {a[0]} is visiting customer {a[1]}')

        return objective_value, variable_values, assignments
    else:
        print('Something went wrong')
        
if __name__ == '__main__':
    random.seed(1)
    Customer_locations = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(3)]
    C = len(Customer_locations)
    OD_locations = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(10)]
    O = len(OD_locations)
    depot_location = [(0,0)]
    total_location_list = depot_location + OD_locations + Customer_locations
    distance = distance_matrix(total_location_list, total_location_list)
    # Detour Matrix
    detour_matrix = np.array([[distance[0,c] + distance[c,o] - distance[0,o] for c in range(1 + O,1 + O + C)] for o in range(1, 1 + O)])
    # Cost of DDs
    cost_of_dedicated_delivery = np.full(C, 7)
    # Arrival probabilities
    individual_arrival_probabilities = np.full(O, 0.5)
    # fixed unavoidable costs
    fixed_negative_utility = 1

    objective_value, variables, assignments = optimize_instance_2D(detour_matrix, cost_of_dedicated_delivery, individual_arrival_probabilities, fixed_negative_utility )

