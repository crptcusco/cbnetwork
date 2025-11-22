# External imports
import logging

from satispy import Variable  # Library to solve SAT problems
from satispy.solver import Minisat  # SAT solver library

# Internal imports
from classes.localscene import LocalAttractor, LocalScene, LocalState
from classes.utils.customtext import CustomText
from classes.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class LocalNetwork:
    def __init__(self, index: int, internal_variables: list):
        """
        Initialize a LocalNetwork instance.

        Args:
            index (int): The index of the local network.
            internal_variables (list): List of internal variables.
        """
        # Manual properties
        self.index = index
        self.internal_variables = internal_variables

        # Processed properties
        self.descriptive_function_variables = []  # List of desired function variables
        self.external_variables = []  # List of external variables
        self.total_variables = []  # List of all variables
        self.total_variables_count = 0  # Total number of variables
        self.cnf_variables_map = {}  # Dictionary of CNF variables

        self.input_signals = []  # List of input signals
        self.output_signals = []  # List of output signals

        # Calculated properties
        self.attractor_count = 1  # Counter for attractors
        self.local_scenes = []  # List of local scenes

    def show(self):
        """
        Display the details of the LocalNetwork instance.
        """
        # Print the subtitle for the local network
        CustomText.make_sub_sub_title(f"Local Network: {self.index}")

        logger = logging.getLogger(__name__)
        logger.info("Internal Variables: %s", self.internal_variables)
        logger.info("External Variables: %s", self.external_variables)
        logger.info("Total Variables: %s", self.total_variables)

        # Print the description of each function variable
        for internal_variable in self.descriptive_function_variables:
            internal_variable.show()

    def process_input_signals(self, input_signals):
        """
        Process the input signals of the local network and update variable lists.

        :param input_signals: List of input signal objects.
        """
        self.input_signals = input_signals
        # Process the input signals and append their indices to the external variables list
        for signal in input_signals:
            self.external_variables.append(signal.index_variable)

        # Add internal variables to the total variables list
        self.total_variables.extend(self.internal_variables.copy())

        # Add external variables to the total variables list
        self.total_variables.extend(self.external_variables.copy())

        # Calculate the number of total variables
        self.total_variables_count = len(self.total_variables)

    def get_internal_variable(self, variable_index):
        """
        Retrieve an internal variable by its index.

        :param variable_index: Index of the internal variable to retrieve.
        :return: The internal variable object if found, otherwise None.
        """
        for internal_variable in self.descriptive_function_variables:
            if internal_variable.index == variable_index:
                return internal_variable
        return None  # Return None if the variable is not found

    def update_internal_variable(self, internal_variable_to_update):
        """
        Update an internal variable in the list of descriptive function variables.

        :param internal_variable_to_update: The internal variable object with updated values.
        """
        for i, internal_variable in enumerate(self.descriptive_function_variables):
            if internal_variable.index == internal_variable_to_update.index:
                self.descriptive_function_variables[i] = internal_variable_to_update
                return True  # Exit after updating the variable
        return False

    @staticmethod
    def evaluate_boolean_function(cnf_function, state, external_values):
        """
        Evaluate a boolean function given the current state and external values.

        :param cnf_function: The boolean function as a string expression (e.g., "(1 and 2) or 3").
                             Variables are represented by their integer indices.
        :param state: Dictionary mapping internal variable indices to their boolean values (0 or 1).
        :param external_values: Dictionary mapping external variable indices to their values.
        :return: The result of the function (0 or 1).
        """
        if not isinstance(cnf_function, str):
            # If it's not a string, we assume it's a CNF list (list of clauses)
            # Example: [[1, -2], [3]] means (1 or not 2) and (3)
            # This is a standard CNF representation where negative numbers mean NOT.
            if isinstance(cnf_function, list):
                # Evaluate CNF
                for clause in cnf_function:
                    clause_satisfied = False
                    for literal in clause:
                        var_index = abs(literal)
                        is_negated = literal < 0

                        # Get value from state or external values
                        if var_index in state:
                            val = state[var_index]
                        elif var_index in external_values:
                            val = external_values[var_index]
                        else:
                            # Default to 0 if not found (should not happen in well-formed networks)
                            val = 0

                        if is_negated:
                            val = 1 - val

                        if val == 1:
                            clause_satisfied = True
                            break

                    if not clause_satisfied:
                        return 0 # One clause false -> entire CNF false
                return 1 # All clauses satisfied
            return 0

        # String evaluation (simple replacement)
        # We need to replace variable indices with their values.
        # To avoid replacing "1" in "10", we should use regex or tokenization.
        # For simplicity in this proof-of-concept, we'll assume variables are tokens.

        expression = cnf_function
        # Replace operators if needed (assuming python syntax for now: and, or, not)
        # If the string uses symbols like ∧, ∨, ~, we need to replace them.
        expression = expression.replace("∧", " and ").replace("∨", " or ").replace("~", " not ")

        # Replace variables
        # We iterate through all known variables (internal + external)
        all_vars = {**state, **external_values}

        # Sort keys by length descending to avoid partial replacement (e.g. replacing 1 in 10)
        # But better is to use regex word boundaries.
        import re

        for var_index, val in all_vars.items():
            # Replace whole word var_index with val
            pattern = r'\b' + str(var_index) + r'\b'
            expression = re.sub(pattern, str(val), expression)

        try:
            result = eval(expression)
            return 1 if result else 0
        except Exception as e:
            logging.getLogger(__name__).error(f"Error evaluating expression '{expression}': {e}")
            return 0

    @staticmethod
    def find_local_attractors_brute_force(local_network, local_scenes=None):
        """
        Find attractors using a brute-force approach by iterating over all possible states.

        :param local_network: The local network object.
        :param local_scenes: List of local scenes (fixed external variable configurations).
        :return: The local network object with updated attractors.
        """
        CustomText.print_simple_line()
        logging.getLogger(__name__).info(
            "FIND ATTRACTORS (BRUTE FORCE) FOR NETWORK: %s", local_network.index
        )

        if local_scenes is None:
            # If no scenes, assume all external variables are 0 (or handle as one empty scene)
            # But usually scenes are required if there are external variables.
            # If no external variables, one 'empty' scene.
            scenes_to_process = [[]]
        else:
            scenes_to_process = local_scenes

        scene_index = 1
        network_attractor_count = 0

        for scene in scenes_to_process:
            # 1. Set up external variables for this scene
            external_values = {}
            if scene:
                for i, val in enumerate(scene):
                    if i < len(local_network.external_variables):
                        ext_var_index = local_network.external_variables[i]
                        external_values[ext_var_index] = int(val)

            # 2. Build State Transition Graph (STG)
            # Iterate over all 2^N states
            num_internal_vars = len(local_network.internal_variables)
            state_map = {} # Map state (tuple) to next state (tuple)

            for i in range(2**num_internal_vars):
                # Convert integer i to binary list representing state
                current_state_vals = [(i >> bit) & 1 for bit in range(num_internal_vars)]
                # Map internal variable indices to values
                current_state_dict = {
                    var: val for var, val in zip(local_network.internal_variables, current_state_vals)
                }

                # Calculate next state
                next_state_vals = []
                for internal_var in local_network.descriptive_function_variables:
                    # Evaluate function for this variable
                    next_val = LocalNetwork.evaluate_boolean_function(
                        internal_var.cnf_function, current_state_dict, external_values
                    )
                    next_state_vals.append(next_val)

                state_map[tuple(current_state_vals)] = tuple(next_state_vals)

            # 3. Detect Cycles (Attractors)
            visited = set()
            scene_attractors = []

            for start_node in state_map:
                if start_node in visited:
                    continue

                path = []
                curr = start_node
                path_set = set()

                while curr not in visited:
                    visited.add(curr)
                    path_set.add(curr)
                    path.append(curr)
                    curr = state_map[curr]

                    if curr in path_set:
                        # Cycle detected
                        cycle_start_index = path.index(curr)
                        cycle = path[cycle_start_index:]

                        # Create LocalAttractor object
                        # We need to convert tuples back to LocalState objects
                        l_states = [LocalState(list(state)) for state in cycle]
                        attractor = LocalAttractor(
                            g_index=0, # Placeholder
                            l_index=len(scene_attractors) + 1,
                            l_states=l_states,
                            network_index=local_network.index,
                            local_scene="".join(scene)
                        )
                        scene_attractors.append(attractor)
                        break

            # 4. Store Attractors
            local_scene_obj = LocalScene(
                scene_index, scene, local_network.external_variables
            )
            local_scene_obj.l_attractors = scene_attractors
            local_network.local_scenes.append(local_scene_obj)

            scene_index += 1
            network_attractor_count += len(scene_attractors)

        local_network.attractor_count = network_attractor_count
        return local_network

    @staticmethod
    def find_local_attractors(local_network, local_scenes=None):
        """
        Find the attractors for a local network.

        :param local_network: The local network object for which to find attractors.
        :param local_scenes: Optional list of local scenes to use for finding attractors.
        :return: The local network object with updated attractors.
        """
        # Print the title for finding attractors
        CustomText.print_simple_line()
        logging.getLogger(__name__).info(
            "FIND ATTRACTORS FOR NETWORK: %s", local_network.index
        )

        if local_scenes is None:
            local_scene = LocalScene(index=1)
            local_scene.l_attractors = LocalNetwork.find_local_scene_attractors(
                local_network=local_network, scene=None
            )
            local_network.local_scenes.append(local_scene)

            local_network.attractor_count = len(local_scene.l_attractors)

        else:
            scene_index = 1
            network_attractor_count = 0
            for scene in local_scenes:
                local_scene = LocalScene(
                    scene_index, scene, local_network.external_variables
                )
                scene_string = "".join(scene)
                local_scene.l_attractors = LocalNetwork.find_local_scene_attractors(
                    local_network=local_network, scene=scene_string
                )
                local_network.local_scenes.append(local_scene)

                # update the scenes index
                scene_index += 1

                # update the attractors index
                network_attractor_count += len(local_scene.l_attractors)

            # Update the count attractor
            local_network.attractor_count = network_attractor_count

        return local_network

    @staticmethod
    def gen_boolean_formulation(
        local_network, n_transitions, attractor_clauses, scene
    ):
        """
        Generate the boolean formulation for the given local network. This formulation includes:
        - CNF variable creation
        - Boolean expressions for each transition
        - Assignment values for permutations
        - Incorporation of attractor clauses

        :param local_network: The local network object for which to generate the boolean formulation.
        :type local_network: LocalNetwork
        :param n_transitions: Number of transitions in the boolean formulation.
        :param attractor_clauses: List of clauses representing attractors.
        :param scene: Optional scene string used to assign specific values to variables.
        :return: The complete boolean function as a Variable object.
        """
        # Create dictionary of CNF variables for each transition
        for variable in local_network.total_variables:
            for transition_c in range(0, n_transitions):
                local_network.cnf_variables_map[f"{variable}_{transition_c}"] = Variable(
                    f"{variable}_{transition_c}"
                )

        transition_count = 0
        boolean_function = Variable("0_0")

        for transition in range(1, n_transitions):
            global_clause_count = 0
            boolean_expression_equivalence = Variable("0_0")

            for variable_model in local_network.descriptive_function_variables:
                clause_count = 0
                boolean_expression_clause_global = Variable("0_0")

                for clause in variable_model.cnf_function:
                    boolean_expression_clause = Variable("0_0")
                    term_count = 0

                    for term in clause:
                        term_aux = abs(int(term))
                        if term_count == 0:
                            if str(term)[0] != "-":
                                boolean_expression_clause = local_network.cnf_variables_map[
                                    f"{term_aux}_{transition - 1}"
                                ]
                            else:
                                boolean_expression_clause = (
                                    -local_network.cnf_variables_map[
                                        f"{term_aux}_{transition - 1}"
                                    ]
                                )
                        else:
                            if str(term)[0] != "-":
                                boolean_expression_clause |= (
                                    local_network.cnf_variables_map[
                                        f"{term_aux}_{transition - 1}"
                                    ]
                                )
                            else:
                                boolean_expression_clause |= (
                                    -local_network.cnf_variables_map[
                                        f"{term_aux}_{transition - 1}"
                                    ]
                                )

                        term_count += 1

                    if clause_count == 0:
                        boolean_expression_clause_global = boolean_expression_clause
                    else:
                        boolean_expression_clause_global &= boolean_expression_clause

                    clause_count += 1

                if global_clause_count == 0:
                    boolean_expression_equivalence = (
                        local_network.cnf_variables_map[
                            f"{variable_model.index}_{transition}"
                        ]
                        >> boolean_expression_clause_global
                    )
                    boolean_expression_equivalence &= (
                        boolean_expression_clause_global
                        >> local_network.cnf_variables_map[
                            f"{variable_model.index}_{transition}"
                        ]
                    )
                else:
                    boolean_expression_equivalence &= (
                        local_network.cnf_variables_map[
                            f"{variable_model.index}_{transition}"
                        ]
                        >> boolean_expression_clause_global
                    )
                    boolean_expression_equivalence &= (
                        boolean_expression_clause_global
                        >> local_network.cnf_variables_map[
                            f"{variable_model.index}_{transition}"
                        ]
                    )

                if not variable_model.cnf_function:
                    logging.getLogger(__name__).warning("ENTER ATYPICAL CASE!!!")
                    boolean_function &= (
                        local_network.cnf_variables_map[
                            f"{variable_model.index}_{transition}"
                        ]
                        | -local_network.cnf_variables_map[
                            f"{variable_model.index}_{transition}"
                        ]
                    )

                global_clause_count += 1

            if transition_count == 0:
                boolean_function = boolean_expression_equivalence
            else:
                boolean_function &= boolean_expression_equivalence

            transition_count += 1

        # Assign values for permutations if a scene is provided
        if scene is not None:
            permutation_count = 0
            for element in local_network.external_variables:
                for v_transition in range(0, n_transitions):
                    if scene[permutation_count] == "0":
                        boolean_function &= -local_network.cnf_variables_map[
                            f"{element}_{v_transition}"
                        ]
                    else:
                        boolean_function &= local_network.cnf_variables_map[
                            f"{element}_{v_transition}"
                        ]

                permutation_count += 1

        # Add attractors to the boolean function
        if attractor_clauses:
            boolean_function_of_attractors = Variable("0_0")
            clause_count = 0

            for clause in attractor_clauses:
                bool_expr_clause_attractors = Variable("0_0")
                term_count = 0

                for term in clause:
                    term_aux = abs(int(term))
                    if term_count == 0:
                        if term[0] != "-":
                            bool_expr_clause_attractors = local_network.cnf_variables_map[
                                f"{term_aux}_{n_transitions - 1}"
                            ]
                        else:
                            bool_expr_clause_attractors = -local_network.cnf_variables_map[
                                f"{term_aux}_{n_transitions - 1}"
                            ]
                    else:
                        if term[0] != "-":
                            bool_expr_clause_attractors &= local_network.cnf_variables_map[
                                f"{term_aux}_{n_transitions - 1}"
                            ]
                        else:
                            bool_expr_clause_attractors &= -local_network.cnf_variables_map[
                                f"{term_aux}_{n_transitions - 1}"
                            ]

                    term_count += 1

                if clause_count == 0:
                    boolean_function_of_attractors = -bool_expr_clause_attractors
                else:
                    boolean_function_of_attractors &= -bool_expr_clause_attractors

                clause_count += 1

            boolean_function &= boolean_function_of_attractors

        # Ensure all variables are included in the boolean function
        for variable in local_network.total_variables:
            boolean_function &= (
                local_network.cnf_variables_map[f"{variable}_0"]
                | -local_network.cnf_variables_map[f"{variable}_0"]
            )

        return boolean_function

    @staticmethod
    def find_local_scene_attractors(local_network, scene=None):
        def count_state_repeat(target_state, path_candidate):
            # input type [[],[],...[]]
            number_of_times = 0
            for element in path_candidate:
                if element == target_state:
                    number_of_times = number_of_times + 1
            return number_of_times

        CustomText.print_simple_line()
        logger.info("Network: %s  Local Scene: %s", local_network.index, scene)

        # First obligatory execution
        set_of_attractors = []
        num_transitions = 3
        attractors = []
        attractor_clauses = []

        # create boolean expression initial with 3 transitions
        boolean_formulation = local_network.gen_boolean_formulation(
            local_network=local_network,
            n_transitions=num_transitions,
            attractor_clauses=attractor_clauses,
            scene=scene,
        )
        sat_response_matrix = []
        # Solve with SAT the boolean formulation
        solver = Minisat()
        solution = solver.solve(boolean_formulation)

        if solution.success:
            for j in range(0, num_transitions):
                sat_response_matrix.append([])
                for i in local_network.total_variables:
                    sat_response_matrix[j].append(
                        solution[local_network.cnf_variables_map[f"{i}_{j}"]]
                    )

        aux_sat_matrix = []
        if len(sat_response_matrix) != 0:
            # TRANSFORM BOOLEAN TO MATRIZ BOOLEAN RESPONSE
            for j in range(0, num_transitions):
                aux_sat_row = []
                for i in range(0, local_network.total_variables_count):
                    if sat_response_matrix[j][i]:
                        aux_sat_row.append("1")
                    else:
                        aux_sat_row.append("0")
                aux_sat_matrix.append(aux_sat_row)
            # m_resp_boolean = aux_sat_matrix
        boolean_response_matrix = aux_sat_matrix
        # BLOCK ATTRACTORS
        # REPEAT CODE

        while len(boolean_response_matrix) > 0:
            path_solution = []
            for path_transition in boolean_response_matrix:
                path_solution.append(path_transition)

            # new list of state attractors
            new_attractor_states = []
            # check attractors
            for state in path_solution:
                state_count = count_state_repeat(state, path_solution)
                if state_count > 1:
                    attractor_begin = path_solution.index(state) + 1
                    attractor_end = path_solution[attractor_begin:].index(state)
                    new_attractor_states = path_solution[
                        attractor_begin - 1 : (attractor_begin + attractor_end)
                    ]
                    attractors = attractors + new_attractor_states
                    # add attractors like list of list
                    set_of_attractors.append(new_attractor_states)
                    break

            # print set_of_attractors
            if len(new_attractor_states) == 0:
                # print ("duplicating")
                num_transitions = num_transitions * 2

            # TRANSFORM LIST OF ATTRACTORS TO CLAUSES
            for clause_attractor in attractors:
                clause_variable = []
                variable_count = 0
                for state_attractor in clause_attractor:
                    if state_attractor == "0":
                        clause_variable.append(
                            "-" + str(local_network.total_variables[variable_count])
                        )
                    else:
                        clause_variable.append(
                            str(local_network.total_variables[variable_count])
                        )
                    variable_count = variable_count + 1
                attractor_clauses.append(clause_variable)

            # print attractor_clauses
            # REPEAT CODE
            boolean_formulation = local_network.gen_boolean_formulation(
                local_network=local_network,
                n_transitions=num_transitions,
                attractor_clauses=attractor_clauses,
                scene=scene,
            )
            sat_response_matrix = []
            solver = Minisat()
            solution = solver.solve(boolean_formulation)

            if solution.success:
                for j in range(0, num_transitions):
                    sat_response_matrix.append([])
                    for i in local_network.total_variables:
                        sat_response_matrix[j].append(
                            solution[local_network.cnf_variables_map[f"{i}_{j}"]]
                        )
            # else:
            #     # print(" ")
            #     print("The expression cannot be satisfied")

            # BLOCK ATTRACTORS
            aux_sat_matrix = []
            if len(sat_response_matrix) != 0:
                # TRANSFORM BOOLEAN TO MATRIZ BOOLEAN RESPONSE
                for j in range(0, num_transitions):
                    aux_sat_row = []
                    for i in range(0, local_network.total_variables_count):
                        if sat_response_matrix[j][i]:
                            aux_sat_row.append("1")
                        else:
                            aux_sat_row.append("0")
                    aux_sat_matrix.append(aux_sat_row)
                # boolean_response_matrix = aux_sat_matrix
            boolean_response_matrix = aux_sat_matrix
            # BLOCK ATTRACTORS
            # REPEAT CODE

        # Creating the objects of the attractor
        scene_attractors = []
        index = 1
        for attractor in set_of_attractors:
            local_states = []
            for state in attractor:
                local_state = LocalState(state)
                local_states.append(local_state)
            local_attractor = LocalAttractor(
                g_index=None,
                l_index=index,
                l_states=local_states,
                network_index=local_network.index,
                relation_index=local_network.external_variables,
                local_scene=scene,
            )

            scene_attractors.append(local_attractor)
            # update the attractors index locally
            index += 1

        logging.getLogger(__name__).info("end find attractors")
        return scene_attractors
