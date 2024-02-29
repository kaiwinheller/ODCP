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
from collections import defaultdict, deque
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment

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
        self.unique_assignment_matrix = np.zeros(detour_matrix.shape)
        self.current_utility_matrix = - self.detour_matrix - fixed_negative_utility
        self.base_utility_matrix = np.copy(self.current_utility_matrix)
        self.current_PAL = list(range(self.C))

    def greedy_inclusion(self):
        '''
        Description: This method aims to find the optimal solution to the ODCP as stated above.
        Iterationwise the unique assignment matrix is extended by a currently not assigned OD
        However, ODs might be indirectly assigned through the assignment of other ODs. This is not reflected in the unique assignment matrix
        In each iteration there are at most 2 customers checked. This is based on the lowest increase necessary to include another OD into the customers range.
        There are however two different solutions that can be different.
        1. a new customer is activated on which an OD is assigned
        2. an already assigned customers compensation is further increased in order to allure more ODs.
        These two possibilities are evaluated and the best is chosen in each iteration
        The function finishes when no increase is observed in each of the 2 cases.

        The optimal solution is updated as attributes so there is no input and no output of the method
        '''
        change_profitable = True
        current_threshold = 0
        while change_profitable:
            change_profitable = False
            UAM_bool = self.current_assignment_matrix.astype(bool)
            # We loop through all customers and find the minimum increase compensation for one assingment shift
            index_list = []
            for c in range(self.C):
                for o in range(self.O):
                    if np.any(UAM_bool[o, :]):
                        currently_assigned_c = np.where(UAM_bool[o, :] == True)[0]
                    else:
                        currently_assigned_c = False
                    if ~UAM_bool[o, c]:
                        UAM_bool_copy = np.copy(UAM_bool)
                        if currently_assigned_c is not False:
                            UAM_bool_copy[o, currently_assigned_c] = False
                        UAM_bool_copy[o, c] = True
                        if self.raise_possible_without_altering_other_assignments(UAM_bool_copy):
                            index_list.append((o, c))
                        else:
                            continue
            for index in range(len(index_list)):
                copy_assign = np.copy(self.unique_assignment_matrix).astype(bool)
                if np.any(copy_assign[index_list[index][0], :]):
                    currently_assigned_c = np.where(copy_assign[index_list[index][0], :] == True)[0]
                    copy_assign[index_list[index][0], currently_assigned_c] = False
                else:
                    currently_assigned_c = False
                copy_assign[index_list[index][0], index_list[index][1]] = True
                true_assignment_matrix, compensations, utils = self.calculate_real_assignment_matrix_from_unique_assignment_matrix(copy_assign) 
                if type(true_assignment_matrix) == str:
                    continue
                expected_savings = self.calculate_expected_savings(np.array(compensations), true_assignment_matrix)
                if expected_savings > current_threshold:
                    change_profitable = True
                    opt_assignment = copy_assign
                    opt_compensation = compensations
                    opt_util = utils
                    current_threshold = expected_savings
                    opt_true_assignment = true_assignment_matrix
            if change_profitable == True:
                self.current_max = current_threshold
                self.current_best_solution = opt_compensation
                print(self.current_best_solution)
                self.unique_assignment_matrix = opt_assignment
                print(self.unique_assignment_matrix.astype(int))
                self.current_utility_matrix = opt_util
                #print(self.current_utility_matrix)
                self.current_assignment_matrix = opt_true_assignment

    def necessary_increase(self, c, utility_mat, UAM_bool):
        '''
        Description: This recursive function calculates the necessary increase in compensation for a certain customer
        in order to maintain the assignments while keeping the compensations at a minimum
        
        Input:
        c:
        type: int
        description: Index of the customer of which compensation is altered

        utility_mat:
        type: np.array() with size (O,C)
        description: The (already altered) Utility matrix

        UAM_bool:
        type: logical np.array() with size (O,C)
        description: This array represents the unique assignment matrix in boolean form
        '''
        # identify the customer that currently would be assigned (if there is any)
        o = np.where(UAM_bool[:, c] == True)[0][0]
        maximum_utility_for_o = np.max(utility_mat[o, :])
        c_2 = np.argmax(utility_mat[o, :])
        o_2 = np.where(UAM_bool[:, c_2] == True)[0][0]
        necessary_inc = maximum_utility_for_o - utility_mat[o, c]
        allowed_inc = utility_mat[o_2, c_2] - utility_mat[o_2, c]
        try:
            if necessary_inc == 0:
                return 0
            elif allowed_inc < necessary_inc:
                return 'Not a valid assignment'
            else:
                return necessary_inc + self.necessary_increase(c_2, utility_mat, UAM_bool)
        except:
            print('This assignment matrix is impossible.')
            return 'Not a valid assignment'

    def raise_possible_without_altering_other_assignments(self, UAM_bool_copy):
        real_assignments, a, b = self.calculate_real_assignment_matrix_from_unique_assignment_matrix(UAM_bool_copy)
        if np.any(real_assignments != UAM_bool_copy):
            return False
        else:
            return True

    def calculate_real_assignment_matrix_from_unique_assignment_matrix(self, UAM):
        '''
        This function calculates the utility matrix from a unique assignment matrix (UAM)
        The UAM represents the assignments that are required for the solution. Other assignments might be 
        unavoidable but the 1s have to stand. The assignment might be impossible. 
        
        Input:
        UAM:
        type: binary np.array() with size (O,C)
        description: Unique assignment matrix

        Output:
        real_assignments = assignment matrix that includes the passive assignments from other ODs (type = np.array(O,C))
        compensations = list of compensations that are needed to fulfill the rules of the assignment_matrix
        utility_mat = Matrix of utilities when using the vector compensations (type = np.array(O,C))
        '''
        UAM_bool = UAM.astype(bool)
        utility_mat = np.copy(self.base_utility_matrix)
        # Move up all utilities so that they match the assignment only looking at the lower bound of 0
        customers_assigned = np.any(UAM == 1, axis=0)
        customers_unassigned = np.all(UAM == 0, axis=0)
        ODs_assigned = np.any(UAM == 1, axis=1)
        ODs_unassigned = np.all(UAM == 0, axis=1)
        # Move up
        increase_vals = {}
        for c in range(self.C):
            if customers_assigned[c]:
                increase_vals[c] = - np.min(self.base_utility_matrix[:, c][UAM_bool[:, c]])
                utility_mat[:, c] = utility_mat[:, c] + increase_vals[c]

        # Loop through all customers assigned
        necessary_increases = {}
        try:
            for c in [p for p in range(self.C) if customers_assigned[p]]:
                necessary_increases[c] = self.necessary_increase(c, utility_mat, UAM_bool)
                utility_mat[:, c] = utility_mat[:, c] + necessary_increases[c]
        except:
            return 'Not possible', None, None

        PAL = self.create_PAL_from_assignment_matrix_and_utility_matrix(UAM_bool, utility_mat, customers_assigned)
        if type(PAL) == str:
            return "Not possible", None, None
        else:
            real_assignments = self.create_full_assignment_matrix(utility_mat, PAL)

        compensations = [0 for c in range(self.C)]
        for c in increase_vals:
            compensations[c] += increase_vals[c]
        for c in necessary_increases:
            compensations[c] += necessary_increases[c]

        return real_assignments, compensations, utility_mat

    def create_PAL_from_assignment_matrix_and_utility_matrix(self, UAM, u, customers_assigned):
        ''' This function creates a PAL from a UAM and a u
        It creates conditions in the form of (customer, [customers that should be preceded by customer])
        for every customer and then in a second step calculates one order that fulfills all conditions (if there is one)
        
        Input:
        UAM:
        type: boolean np.array() with size(O,C) that represents the unique assignment matrix
        description: that represents the unique assignment matrix
        
        u:
        type: np.array() with size (O,C)
        description: The utility matrix that fits to the UAM
        
        customers_assigned
        type: boolean np.array() with size C
        description: Boolean array that is True if at least one customer is already assigned to c
        
        
        Output:
        PAL_order:
        type: list
        description: The priority assignment order list that is a tiebreaker if the utility is equal'''
        conditions = []
        for c in [p for p in range(self.C) if customers_assigned[p]]:
            o = np.where(UAM[:, c] == True)[0][0]
            util = u[o,c]
            logical = u[o, :] == util
            current_list = [c, []]
            for cs in np.arange(self.C)[logical]:
                if cs != c:
                    current_list[1].append(cs)
            conditions.append(current_list)
        PAL_order = self.find_order(self.C, conditions)
        return PAL_order

    def find_order(self, n, conditions):
        '''Description: This function finds the fitting order that fulfills all the conditions specified in conditions
        
        Input: 
        n:
        type: integer
        description: Number of customers
        
        conditions:
        type: list
        description: List that as a first entry has the customer index and as second entry a list with all customers that should be a lower priority
        
        Outputs:
        order:
        type: list
        description: A valid order that serves as priority assignment list (PAL)'''

        # Build the graph with reversed edges based on the new requirement
        graph = defaultdict(list)
        indegree = {i: 0 for i in range(n)}  # In-degree for all nodes
        for num, successors in conditions:
            for succ in successors:
                graph[num].append(succ)
                indegree[succ] += 1

        # Perform topological sort
        queue = deque([node for node in indegree if indegree[node] == 0])
        order = []

        while queue:
            node = queue.popleft()
            order.append(node)
            for neighbor in graph[node]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) == n:
            return order
        else:
            return "Invalid ordering"  # Cycle detected or not all nodes covered   

    def create_full_assignment_matrix(self, Utility_Matrix, PAL):
        '''Description: This function creates the full assignment matrix out of a PAL and a Utility_Matrix
        The first condition that the assignment is based on is the utility. If the utility ties, the PAL is used as a tiebreaker
        customers that have a lower index are prioritized over higher index customers.
        
        Input:
        Utility_Matrix:
        type: np.array with size (O,C)
        description: Final utility matrix
        
        PAL:
        type: list
        description: List that the represents an order of preference for the customers
        One could interpret that as every customer to the left has a tiny higher compensation than the higher indexed customers on the right.
        
        Output:
        assignment_matrix_copy:
        type: np.array() with size (O,C)
        description: the real assignment matrix with the passive assignments included'''
        # Reassign os based on updated utility matrix and PAL
        assignment_matrix_copy = np.zeros((self.O, self.C))
        for o in range(O):
            # Find c with the highest utility for each o, considering PAL for ties
            preferences = np.argsort(-Utility_Matrix[o, :])  # Sort cs by decreasing utility for o
            preferences = sorted(preferences, key=lambda x: (-Utility_Matrix[o, x], PAL.index(x)))
            best_c = preferences[0]
            assignment_matrix_copy[o, :] = 0  # Remove o from all cs
            if Utility_Matrix[o, preferences[0]] >= 0:
                assignment_matrix_copy[o, best_c] = 1  # Assign o to the best c
            else:
                pass
        return assignment_matrix_copy

# ________________________________ Not Used as of NOW ________________________________________________


    def increase_compensation_for_c_till_new_inclusion(self, c):
        assignment_matrix_copy = np.copy(self.current_assignment_matrix)
        compensations_copy = np.copy(self.current_best_solution)
        utility_matrix_copy = np.copy(self.current_utility_matrix)
        PAL_copy = copy(self.current_PAL)
        
        # Identify the OD that is added to the mix
        ODs_without_ones = np.all(assignment_matrix_copy == 0, axis=1)
        # Apply the logical array to mask the column, then find the index of the max value in that column
        max_index = np.argmax(utility_matrix_copy[ODs_without_ones, c])
        # The result is directly in terms of the original indices where the logical condition is True
        o = np.where(ODs_without_ones)[0][max_index]  # Map to the original row index
        increase_level = assignment_matrix_copy[o, c] - compensations_copy[c]

        # Check if there is a switch in assignment for other ODs associated with that assignment
        ODs_with_ones = ~ODs_without_ones
        new_utility_for_c = utility_matrix_copy[ODs_with_ones, c]


        # Calculate new compensation for c based on utility differences and current assignments
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
        if currently_assigned_c.size > 0 and utility_matrix_copy[o, c] == utility_matrix_copy[o, currently_assigned_c]:
            # Update the PAL-list
            index_to_reassign = 0 #PAL_copy.index(currently_assigned_c)
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

        # Calculate new compensation for c based on utility differences and current assignments
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
        if currently_assigned_c.size > 0 and utility_matrix_copy[o, c] == utility_matrix_copy[o, currently_assigned_c]:
            # Update the PAL-list
            index_to_reassign = 0 #PAL_copy.index(currently_assigned_c)
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
    
    def decrease_compensation_for_c_till_o_joins(self, c, o):
        assignment_matrix_copy = np.copy(self.current_assignment_matrix)
        compensations_copy = np.copy(self.current_best_solution)
        utility_matrix_copy = np.copy(self.current_utility_matrix)
        PAL_copy = copy(self.current_PAL)
        if assignment_matrix_copy[o, c] == 0:
            return compensations_copy, assignment_matrix_copy, utility_matrix_copy, False
        
        # Compute the maximum utility of all ODs associated with c that are assigned to c
        mask = (assignment_matrix_copy[:, c] == 1) & (utility_matrix_copy[:, c] < utility_matrix_copy[o, c])

        # Identify what the necessary decrease amount is
        if np.any(mask):
            utility_of_lower_relevant_OD = np.max(utility_matrix_copy[:, c][mask], axis=0)
            required_decrease = utility_matrix_copy[o, c] - utility_of_lower_relevant_OD
        else:
            required_decrease = utility_matrix_copy[o, c]

        # Calculate new compensation for c based on required_decrease
        if required_decrease > 0:
            improvement = True
        else:
            improvement = False

        # Update compensation for c
        compensations_copy[c] = compensations_copy[c] - required_decrease

        # Update utility matrix for c
        utility_matrix_copy[:, c] -= required_decrease
        
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





            


