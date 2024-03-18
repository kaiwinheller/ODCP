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
from itertools import permutations
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

    def create_divide_and_counquer_solution(self):
        '''
        This function outputs the best soltion for the case in which all customers are viewed separately
        '''
        compensations = np.full(self.C, 0).astype(float)
        for c in range(self.C):
            expected_savings_now = 0
            for o in sorted(range(self.O), key = lambda d: - self.base_utility_matrix[d,c]):
                expected_savings_when_including_o = self.expected_savings_when_compensation_is_raised_in_order_to_include_o(o, c)
                if expected_savings_when_including_o > expected_savings_now:
                    compensations[c] = - self.base_utility_matrix[o,c]
                    expected_savings_now = expected_savings_when_including_o
                else:
                    break

        return compensations

    def expected_savings_when_compensation_is_raised_in_order_to_include_o(self, o, c):
        '''
        This function checks, whether increasing the compensation for c results in a savings increase
        if all network-effects are neglected
        '''
        compensation = - self.base_utility_matrix[o,c]
        savings_generated_if_served = self.cost_of_dedicated_delivery[c] - compensation
        utilities_customer = self.base_utility_matrix[:, c] + compensation
        ODs_motivated = np.where(utilities_customer >= 0)[0]
        serving_probability = 1 - np.product(([1 - individual_arrival_probabilities[ODs_motivated]]))
        expected_savings = serving_probability * savings_generated_if_served
        return expected_savings
    
    def neighborhood_search_lower_one_compensation(self, solution):
        solution_list = []
        for c in range(self.C):
            increased_compensation_solution = self.get_lowered_compensation_solution_for_customer_c(c, np.copy(solution))
            solution_list.append(increased_compensation_solution)
        return solution_list

    def neighborhood_search_lower_one_compensation_necessary(self, solution):
        solution_list = []
        for c in range(self.C):
            increased_compensation_solution = self.get_necessary_lowered_compensation_solution_for_customer_c(c, np.copy(solution))
            solution_list.append(increased_compensation_solution)
        return solution_list
    
    def neighborhood_search_higher_one_compensation_necessary(self, solution):
        solution_list = []
        for c in range(self.C):
            increased_compensation_solution = self.get_necessary_increased_compensation_solution_for_customer_c(c, np.copy(solution))
            solution_list.append(increased_compensation_solution)
        return solution_list
    
    def neighborhood_search_lower_one_compensation_withouth_harm(self, solution):
        solution_list = []
        for c in range(self.C):
            increased_compensation_solution = self.get_decrease_without_harm_solution_for_customer_c(c, np.copy(solution))
            solution_list.append(increased_compensation_solution)
        return solution_list
    

    
    def get_increased_compensation_solution_for_customer_c(self, c, solution):
        utility_vector_for_c = self.base_utility_matrix[:, c] + solution[c]
        try:
            compensation_increase_needed = - max(utility_vector_for_c[utility_vector_for_c < 0])
        except ValueError:
            compensation_increase_needed = 0
        solution[c] = solution[c] + compensation_increase_needed
        return solution
    
    def neighborhood_search_upper_one_compensation(self, solution):
        solution_list = []
        for c in range(self.C):
            increased_compensation_solution = self.get_increased_compensation_solution_for_customer_c(c, np.copy(solution))
            solution_list.append(increased_compensation_solution)
        return solution_list
    
    def extract_best_solution(self, found_a_better_solution, best_solution, best_assignment, expected_savings_now):
        best_solution_index = np.argmax(expected_savings_now)
        best_solution_of_the_search = best_solution[best_solution_index]
        best_assignment_of_the_search = best_assignment[best_solution_index]
        best_objective_value = expected_savings_now[best_solution_index]
        return found_a_better_solution, best_solution_of_the_search, best_assignment_of_the_search, best_objective_value
    
    def neighborhood_search(self, solution):
        expected_savings_now, best_assignment = self.expected_savings_for_a_solution(solution)
        best_solution = np.copy(solution)
        found_a_better_solution = False
        # Hier wird die gesamte Nachbarschaft erstellt (Die Funktionen sollten hier die solutions wiedergeben)
        neighborhood_functions = [self.neighborhood_search_lower_one_compensation(np.copy(solution)),
                                  self.neighborhood_search_upper_one_compensation(np.copy(solution)),
                                  self.neighborhood_search_lower_one_compensation_withouth_harm(np.copy(solution)),
                                  self.neighborhood_search_higher_one_compensation_necessary(solution),
                                  self.neighborhood_search_lower_one_compensation_necessary(solution)]
        neighborhood_list = [item for sublist in neighborhood_functions for item in sublist]
        for sol in neighborhood_list:
            expected_savings_for_solution, assignment = self.expected_savings_for_a_solution(sol)
            if expected_savings_for_solution > expected_savings_now:
                found_a_better_solution = True
                best_solution = sol
                best_assignment = assignment
                expected_savings_now = expected_savings_for_solution
        return found_a_better_solution, best_solution, best_assignment, expected_savings_now

    def iterative_decrease_a_solution_by_best_fit(self, solution):
        '''
        This function takes in a solution in form of a compensation array and iteratively improves it by
        decreasing single compensations in order to exclude one of the ODs out of accepting range.
        '''
        while True:
            expected_savings_now, _ = self.expected_savings_for_a_solution(solution)
            found_a_better_solution, solution_found, assignment, objective_value = self.neighborhood_search(solution)
            if found_a_better_solution:
                solution = solution_found
            else:
                return solution

    def get_lowered_compensation_solution_for_customer_c(self, c, solution):
        '''
        This method returns the solution that materializes as the compensation for c is decreased
        as much as needed for the next OD(s) to be excluded'''
        utility_vector_for_c = self.base_utility_matrix[:, c] + solution[c]
        try:
            compensation_decrease_needed = min(utility_vector_for_c[utility_vector_for_c > 0])
        except ValueError:
            compensation_decrease_needed = solution[c]
        solution[c] = solution[c] - compensation_decrease_needed
        return solution
    
    def get_utility_mat(self, solution):
        return self.base_utility_matrix + solution.T

    def difference_between_top_entries(self, V, B):
        """
        Finds the difference between the highest and second highest (or third if needed) entries
        in V that are greater or equal to 0, based on a filtering provided by B.
        
        Args:
        - V (numpy.ndarray): A vector of values.
        - B (numpy.ndarray): A boolean vector indicating which elements of V to consider.
        
        Returns:
        - float or None: The difference as specified or None if not enough entries above 0.
        """
        # Filter V by B and then select entries >= 0
        filtered_V = V[B]
        eligible_values = filtered_V[filtered_V >= 0]
        
        # Check if there are less than 2 eligible values
        if eligible_values.size < 2:
            return None
        
        # Sort eligible values in descending order
        sorted_values = np.sort(eligible_values)[::-1]
        
        # Find the difference according to the rules
        if sorted_values[0] != sorted_values[1]:
            return sorted_values[0] - sorted_values[1]
        else:
            # If the first two are equal, look for a third distinct value
            for i in range(2, len(sorted_values)):
                if sorted_values[i] < sorted_values[0]:  # Find the next distinct value
                    return sorted_values[0] - sorted_values[i]
            # If no third distinct value found
            return None
        
    def get_necessary_lowered_compensation_solution_for_customer_c(self, c, solution):
        '''
        This method returns the solution that materializes as the compensation for c is decreased
        as much as needed for the next OD(s) to be excluded 
        The difference from the other function is, that this time the 
        utility is not decreased to 0 as much as possible in order to not
        affect any other assignments'''
        utility_mat = self.get_utility_mat
        solution_copy = np.copy(solution)
        try:
            # Maximum Utility Vector
            maximum_utility_vector = np.max(utility_mat, axis= 1)
            # c-th column of the utility mat
            c_column = utility_mat[:,c]
            # Differences of the two
            differences_vector = c_column - maximum_utility_vector
            # Get the binary vector with differences
            differences_vector_binary = differences_vector[differences_vector >= 0]
            # Check how many possible assignments are made
            number_possible_assignments = np.count_nonzero(differences_vector_binary)
            # Check wether there is one assignment
            if number_possible_assignments == 1:
                solution_copy[c] = 0
                return solution_copy
            elif number_possible_assignments == 0:
                return solution_copy
            elif number_possible_assignments > 1:
                # Identify the highest and the second highest difference
                decrease = self.difference_between_top_entries(differences_vector, differences_vector_binary)
                solution_copy[:,c] -= decrease
                return solution_copy
        except:
            return solution_copy
    
    def get_decrease_without_harm_solution_for_customer_c(self, c, solution):
        utility_mat = self.get_utility_mat
        solution_copy = np.copy(solution)
        try:
            # Maximum Utility Vector
            maximum_utility_vector = np.max(utility_mat, axis= 1)
            # c-th column of the utility mat
            c_column = utility_mat[:,c]
            # Differences of the two
            differences_vector = c_column - maximum_utility_vector
            # Get the binary vector with differences
            differences_vector_binary = differences_vector[differences_vector >= 0]
            # Check how many possible assignments are made
            number_possible_assignments = np.count_nonzero(differences_vector_binary)
            # Check whether there is one assignment
            if number_possible_assignments == 1:
                no_harm_reduction = differences_vector[differences_vector_binary]
                solution_copy[:, c] -= no_harm_reduction
                return solution_copy
            elif number_possible_assignments == 0:
                return solution_copy
            elif number_possible_assignments > 1:
                # Identify the highest and the second highest difference
                no_harm_reduction = np.min(differences_vector[differences_vector_binary])
                solution_copy[:,c] -= no_harm_reduction
                return solution_copy
        except:
            return solution_copy

    def get_necessary_increased_compensation_solution_for_customer_c(self, c, solution):
        '''
        This method returns the solution that materializes as the compensation for c is decreased
        as much as needed for the next OD(s) to be excluded 
        The difference from the other function is, that this time the 
        utility is not decreased to 0 as much as possible in order to not
        affect any other assignments'''
        utility_mat = self.get_utility_mat
        solution_copy = np.copy(solution)
        try:
            # Maximum Utility Vector
            maximum_utility_vector = np.max(utility_mat, axis= 1)
            # c-th column of the utility mat
            c_column = utility_mat[:,c]
            # Differences of the two
            differences_vector = c_column - maximum_utility_vector
            # Get the binary vector with differences
            differences_vector_binary = differences_vector[differences_vector < 0]
            # Check how many possible assignments are made
            minimum_necessary_increase = np.max(differences_vector[differences_vector_binary])
            solution_copy[:, c] -= minimum_necessary_increase
            return solution_copy
        except:
            return solution_copy
    
    def create_binary_matrix(self, U):
        max_values = np.max(U, axis=1)
        mask_positive = U >= 0
        mask_max = U == max_values[:, np.newaxis]
        A = mask_positive & mask_max
        return A.astype(int)
    
    def extract_ODs_that_are_indifferent(self, matrix):
        mask = (matrix > 0).sum(axis=1) > 1
        ODs = np.where(mask)[0]
        return ODs
    
    def get_column_indices(self, ODs, A):
        """
        Get column indices that have at least one occurrence of 1 in any of the specified rows.

        Args:
        ODs (list): List of indices of rows.
        A (numpy.ndarray): Binary matrix.

        Returns:
        list: List of column indices.
        """
        # Extract specified rows from binary matrix A
        specified_rows = A[ODs]

        # Find column indices where there is at least one occurrence of 1 in any of the specified rows
        column_indices = np.where(np.any(specified_rows == 1, axis=0))[0]

        return column_indices
    
    def adjust_the_assignments_according_to_the_PAL(self, PAL, assignments):
        updated_assignments = assignments.copy()

        # Iterate through indices in preference order
        for count, idx in enumerate(PAL):
            # Check if the current index is already assigned
            if assignments[idx] == 1:
                # Update indices with lower index to 0 if they are currently unassigned
                for i in PAL[(count+1):]:
                    if updated_assignments[i] == 1:
                        updated_assignments[i] = 0
                break

        return updated_assignments
    
    def get_expected_savings_from_assignment_and_solution(self, assignment_matrix, solution):
        expected_savings = 0
        for c in range(self.C):
            serving_probability = 1 - np.product(([1 - self.individual_arrival_probabilities[np.where(assignment_matrix[:, c] == 1)]]))
            savings_if_served = self.cost_of_dedicated_delivery[c] - solution[c]
            expected_savings += serving_probability * savings_if_served

        return expected_savings
    
    def expected_savings_for_a_solution(self, solution):
        '''
        This method returns the expected savings that results form a solution
        Sometimes the problem is a Tie-Breaking when the utility for two or more customers is equal
        for an OD. in this case we can increase the compensation of a customer a little tiny bit 
        (by a small invisible fraction), that is represented by a priority allocation list (PAL)
        This might be terribly slow in some instances but we might get it working.
        '''
        solution = solution
        utility_matrix_now = np.around(self.base_utility_matrix + solution, 5) # Here is a potential error since I don't know in which dimension its added
        base_assignment_matrix = self.create_binary_matrix(utility_matrix_now)

        # Handling of the case where one ore more ODs are indifferent
        ODs_that_are_indifferent = self.extract_ODs_that_are_indifferent(base_assignment_matrix)
        customers_that_are_affected = self.get_column_indices(ODs_that_are_indifferent, base_assignment_matrix)
        if list(ODs_that_are_indifferent):
            list_of_permutations_of_the_affected_customers = list(permutations(customers_that_are_affected))
            best_expected_savings_so_far = 0
            for permutation in list_of_permutations_of_the_affected_customers:
                assignment_matrix = np.copy(base_assignment_matrix)
                for o in ODs_that_are_indifferent:
                    assignment_matrix[o, :] = self.adjust_the_assignments_according_to_the_PAL(permutation, base_assignment_matrix[o, :])
                expected_savings_for_that_assignment = self.get_expected_savings_from_assignment_and_solution(assignment_matrix, solution)
                if expected_savings_for_that_assignment > best_expected_savings_so_far:
                    best_assignment = assignment_matrix
                    best_expected_savings_so_far = expected_savings_for_that_assignment
        else:
            best_assignment = base_assignment_matrix
            best_expected_savings_so_far = self.get_expected_savings_from_assignment_and_solution(best_assignment, solution)

        return best_expected_savings_so_far, best_assignment

# --------------- Not included ------------------- #

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


if __name__ == '__main__':
    random.seed(9)
    Customer_locations = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(3)]
    C = len(Customer_locations)
    OD_locations = [(random.uniform(-10, 10), random.uniform(-10, 10)) for _ in range(3)]
    O = len(OD_locations)
    depot_location = [(0,0)]
    total_location_list = depot_location + OD_locations + Customer_locations
    distance = distance_matrix(total_location_list, total_location_list)
    # Detour Matrix
    detour_matrix = np.array([[distance[0,c] + distance[c,o] - distance[0,o] for c in range(1 + O,1 + O + C)] for o in range(1, 1 + O)])
    '''
    detour_matrix = np.array([
        [20, 17, 16.5],
        [21, 16, 19],
        [19, 15, 18]
    ])
    '''
    # Cost of DDs
    cost_of_dedicated_delivery = np.full(C, 30)
    # Arrival probabilities
    individual_arrival_probabilities = np.full(O, 0.5)
    # fixed unavoidable costs
    fixed_negative_utility = 1

    Instance = branch_and_bound_ODCP(detour_matrix, cost_of_dedicated_delivery, individual_arrival_probabilities, fixed_negative_utility)
    solution = Instance.create_divide_and_counquer_solution()
    print(f'Initial solution: {solution} Objective value: {Instance.expected_savings_for_a_solution(solution)[0]}')

    solution = Instance.iterative_decrease_a_solution_by_best_fit(solution)
    print(f'Enhanced solution: {solution} Objective value: {Instance.expected_savings_for_a_solution(solution)[0]}')










            


