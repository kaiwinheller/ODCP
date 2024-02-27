# This file contains a function that uses a branch-and-bound method to solve any ODCP with the following characteristics
# Shape : arbitrary
# depot : arbitrary
# distance metric : arbitrary
# OD Locations : discrete
# OD arrivals : simultanious
# compensation : static
# OD arrivals : potentially many
# customers : potentially many

import numpy as np
import random
from copy import copy
from scipy.spatial import distance_matrix

class branch_and_bound_ODCP():
    def __init__(self, detour_matrix, cost_of_dedicated_delivery, individual_arrival_probabilities, fixed_negative_utility = 0):
        self.C = detour_matrix.shape[1]
        self.O = detour_matrix.shape[0]
        self.detour_matrix = np.around(detour_matrix, 4)
        self.cost_of_dedicated_delivery = cost_of_dedicated_delivery
        self.individual_arrival_probabilities = individual_arrival_probabilities
        self.fixed_negative_utility = fixed_negative_utility
        self.current_max = 0
        self.current_best_solution = np.zeros(self.C)
        self.current_assignment_matrix = np.zeros(detour_matrix.shape)
        self.current_utility_matrix = - self.detour_matrix - fixed_negative_utility
        self.current_PAL = list(range(self.C))
        '''
        # Dictionary that includes customers as keys and a sorted list of ODs as values. The sorting criterium is the lowest detour or in other words
        # the ODs that require the least compensation in order for them to participate in the delivery
        self.priority_list_dict = {c: sorted(range(O), key= lambda o: detour_marix[o, c]), for c in range(C)}
        # Dictionary that includes ODs as keys and a list of compensations as values. The compensations are the minimum compensation required for the delivery to
        # cutomer i
        self.participation_compensation = {o: [detour_matrix[o, c] + fixed_negative_utility for c in range(C)] for i in range(O)}
        # A sorted list, which includes all customers. The sorting criterium is the lowest detour from any OD in ascending order.
        self.delivery_location_order = sorted(range(C), key = lambda c: detour_matrix[self.priority_list_dict[c][0], c])
        # Contains a list of ODs that are fixed
        self.fixed_ODs = []
        '''

    def greedy_inclusion(self):
        increase_possible = True
        while increase_possible:
            increase_possible = False
            greedy_maximum = self.current_max
            for c in range(self.C):
                for o in range(self.O):
                    # increase the compensation for 1 c
                    compensation_vector, assignment_matrix, utility_matrix_copy, improvement = self.increase_compensation_for_c_till_o_joins(c, o)
                    # Calculate the expected savings
                    if improvement:
                        expected_savings = self.calculate_expected_savings(compensation_vector, assignment_matrix)
                    else:
                        continue
                    if expected_savings > greedy_maximum:
                        greedy_maximum = expected_savings
                        best_compensation_vector = compensation_vector
                        best_assignment_matrix = assignment_matrix
                        best_utility_matrix = utility_matrix_copy
                        increase_possible = True
            if increase_possible == True:
                self.current_max = greedy_maximum
                self.current_best_solution = best_compensation_vector
                print(self.current_best_solution)
                self.current_assignment_matrix = best_assignment_matrix
                print(self.current_assignment_matrix)
                self.current_utility_matrix = best_utility_matrix
                print(self.current_utility_matrix)

    def increase_compensation_for_c_till_o_joins(self, c, o):
        assignment_matrix_copy = np.copy(self.current_assignment_matrix)
        compensations_copy = np.copy(self.current_best_solution)
        utility_matrix_copy = np.copy(self.current_utility_matrix)
        PAL_copy = copy(self.current_PAL)
        if assignment_matrix_copy[o, c] == 1:
            return compensations_copy, assignment_matrix_copy, utility_matrix_copy, False
        
        # Compute the maximum of 0 and each element in the row-wise max of utility_matrix_copy
        first_term = np.maximum(0, np.max(utility_matrix_copy, axis=1))

        # Subtract utility_matrix_copy[:, c] from this first_term
        utility_diffs = first_term - utility_matrix_copy[:, c]

        min_required_increase = 1000
        # Calculate new compensation for c based on utility differences and current assignments
        improvement = False
        currently_assigned_c = np.where(assignment_matrix_copy[o, :] == 1)[0]
        increase_level = np.around(utility_diffs[o], 4)
        if increase_level < 0:
            improvement = False
        else:
            improvement = True

        # Update compensation for c
        compensations_copy[c] = compensations_copy[c] + increase_level
        # Update utility matrix for c
        utility_matrix_copy[:, c] += (compensations_copy[c] - self.current_best_solution[c])
        if currently_assigned_c.size > 0 and increase_level == utility_diffs[currently_assigned_c[0]]:
            # Update the PAL-list
            index_to_reassign = PAL_copy.index(currently_assigned_c)
            PAL_copy.remove(c)
            PAL_copy.insert(index_to_reassign, c)
        if increase_level < 0:
            return compensations_copy, assignment_matrix_copy, utility_matrix_copy, improvement
        
        # Reassign os based on updated utility matrix and PAL
        for o in range(O):
            # Find c with the highest utility for each o, considering PAL for ties
            preferences = np.argsort(-utility_matrix_copy[o, :])  # Sort cs by decreasing utility for o
            preferences = sorted(preferences, key=lambda x: (-utility_matrix_copy[o, x], PAL_copy.index(x)))
            best_c = preferences[0]
            assignment_matrix_copy[o, :] = 0  # Remove o from all cs
            if utility_matrix_copy[o, preferences[0]] >= 0:
                assignment_matrix_copy[o, best_c] = 1  # Assign o to the best c
            else:
                pass
        
        return compensations_copy, assignment_matrix_copy, utility_matrix_copy, improvement

    def calculate_expected_savings(self, compensation_vector, assignment_matrix):
        '''
        Description: The function calculates the savings (which are to maximize) of a certain policy.
        The policy is represented by an assignment matrix, which assigns a customer to an OD and 
        a compensation vector, which holds compensations for every customer.

        Parameters:
        compensation_vector:
        Type: np.array() (size C)
        Description: Vector of compensations

        assignment_matrix:
        Type: np.array() (size (O,C))
        Description: Binary matrix with 1 indicating that an OD is assigned to a customer and 0 if not

        Output:
        expected_savings:
        Type: float
        Description: the expected savings considering the policy
        '''
        # Calculate the serving probability
        no_show_probability_matrix = 1 - assignment_matrix * self.individual_arrival_probabilities[:, np.newaxis]

        # Calculate the probability that a customer is served
        serving_probability = 1 - np.prod(no_show_probability_matrix, axis=0)

        # calculate the expected costs
        expected_savings = np.matmul(serving_probability, (self.cost_of_dedicated_delivery - compensation_vector).T)

        return expected_savings
        


if __name__ == '__main__':
    random.seed(7)
    Customer_locations = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(3)]
    C = len(Customer_locations)
    OD_locations = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(3)]
    O = len(OD_locations)
    depot_location = [(0,0)]
    total_location_list = depot_location + OD_locations + Customer_locations
    distance = distance_matrix(total_location_list, total_location_list)
    # Detour Matrix
    detour_matrix = np.array([[distance[0,c] + distance[c,o] - distance[0,o] for c in range(1 + O,1 + O + C)] for o in range(1, 1 + O)])
    # Cost of DDs
    cost_of_dedicated_delivery = np.full(C, 1000)
    # Arrival probabilities
    individual_arrival_probabilities = np.full(O, 0.5)
    # fixed unavoidable costs
    fixed_negative_utility = 1

    Instance = branch_and_bound_ODCP(detour_matrix, cost_of_dedicated_delivery, individual_arrival_probabilities, fixed_negative_utility )
    Instance.greedy_inclusion()





            


