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

class branch_and_bound_ODCP():
    def __init__(self, detour_matrix, cost_of_dedicated_delivery, individual_arrival_probabilities, fixed_negative_utility = 0):
        self.C = detour_matrix.shape()[1]
        self.O = detour_matrix.shape()[0]
        self.detour_matrix = detour_matrix
        self.cost_of_dedicated_delivery = cost_of_dedicated_delivery
        self.individual_arrival_probabilities = individual_arrival_probabilities
        self.fixed_negative_utility = fixed_negative_utility
        self.current_max = 0
        self.current_best_solution = np.zeros(self.C)
        self.current_assignment_matrix = np.zeros(detour_matrix.shape())
        self.current_utility_matrix = - detour_matrix - fixed_negative_utility
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
                # increase the compensation for 1 c
                compensation_vector, assignment_matrix, utility_matrix_copy, PAL = self.increase_compensation_for_c(c)
                # Calculate the expected savings
                expected_savings = self.calculate_expected_savings(compensation_vector, assignment_matrix)
                if expected_savings > greedy_maximum:
                    greedy_maximum = expected_savings
                    best_compensation_vector = compensation_vector
                    best_assignment_matrix = assignment_matrix
                    increase_possible = True
            if increase_possible == True:
                self.current_max = greedy_maximum
                self.current_best_solution = best_compensation_vector
                self.current_assignment_matrix = best_assignment_matrix

    def increase_compensation_for_c(self, c):
        assignment_matrix_copy = self.current_assignment_matrix
        compensations_copy = self.current_best_solution
        utility_matrix_copy = self.current_utility_matrix
        PAL_copy = self.current_PAL
        
        # Calculate the difference in utility needed for each o to prefer c over its current assignment
        utility_diffs = np.max(utility_matrix_copy, axis=1) - utility_matrix_copy[:, c]
        min_required_increase = 1000
        # Calculate new compensation for c based on utility differences and current assignments
        for o in range(self.O):
            if assignment_matrix_copy[o, c] == 0:  # o is not assigned to c
                required_increase = utility_diffs[o]
                if required_increase < min_required_increase:
                    increase_level = np.around(required_increase,4)
                    currently_assigned_c = np.where(assignment_matrix_copy[o, :] == 1)[0]

        if increase_level > 0:    
            # Update compensation for c
            compensations_copy[c] = compensations_copy[c] + increase_level
            # Update utility matrix for c
            utility_matrix_copy[:, c] += (compensations_copy[c] - self.current_best_solution[c])
        elif increase_level == 0:
            # Update the PAL-list
            index_to_reassign = PAL_copy.index(currently_assigned_c)
            PAL_copy.remove(c)
            PAL_copy.insert(index_to_reassign, c)

        
        # Reassign os based on updated utility matrix and PAL
        for o in range(O):
            # Find c with the highest utility for each o, considering PAL for ties
            preferences = np.argsort(-utility_matrix_copy[o, :])  # Sort cs by decreasing utility for o
            preferences = sorted(preferences, key=lambda x: (-utility_matrix_copy[o, x], PAL_copy.index(x)))
            best_c = preferences[0]
            assignment_matrix_copy[o, :] = 0  # Remove o from all cs
            assignment_matrix_copy[o, best_c] = 1  # Assign o to the best c
        
        return compensations_copy, assignment_matrix_copy, utility_matrix_copy, PAL_copy



            


