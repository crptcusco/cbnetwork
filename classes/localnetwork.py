# External imports
from satispy import Variable  # Library to solve SAT problems
from satispy.solver import Minisat  # SAT solver library

# Internal imports
from classes.localscene import LocalScene, LocalAttractor, LocalState
from classes.utils.customtext import CustomText


class LocalNetwork:
    def __init__(self, num_local_network, l_var_intern):
        """
        Initialize a LocalNetwork instance.

        Args:
            num_local_network (int): The index of the local network.
            l_var_intern (list): List of internal variables.
        """
        # Manual properties
        self.index = num_local_network
        self.l_var_intern = l_var_intern

        # Processed properties
        self.des_funct_variables = []  # List of desired function variables
        self.l_var_exterm = []  # List of external variables
        self.l_var_total = []  # List of all variables
        self.num_var_total = 0  # Total number of variables
        self.dic_var_cnf = {}  # Dictionary of CNF variables

        self.l_input_signals = []  # List of input signals
        self.l_output_signals = []  # List of output signals

        # Calculated properties
        self.count_attractor = 1  # Counter for attractors
        self.l_local_scenes = []  # List of local scenes

    def show(self):
        """
        Display the details of the LocalNetwork instance.
        """
        # Print the subtitle for the local network
        CustomText.make_sub_sub_title(f"Local Network: {self.index}")

        # Print the internal variables
        print('Internal Variables: ', self.l_var_intern)

        # Print the external variables
        print('External Variables: ', self.l_var_exterm)

        # Print the total variables
        print('Total Variables: ', self.l_var_total)

        # Print the description of each function variable
        for o_internal_variable in self.des_funct_variables:
            o_internal_variable.show()

    def process_input_signals(self, l_input_signals):
        """
        Process the input signals of the local network and update variable lists.

        :param l_input_signals: List of input signal objects.
        """
        # Process the input signals and append their indices to the external variables list
        for o_signal in l_input_signals:
            self.l_var_exterm.append(o_signal.index_variable)

        # Add internal variables to the total variables list
        self.l_var_total.extend(self.l_var_intern.copy())

        # Add external variables to the total variables list
        self.l_var_total.extend(self.l_var_exterm.copy())

        # Calculate the number of total variables
        self.num_var_total = len(self.l_var_total)

    def get_internal_variable(self, i_variable):
        """
        Retrieve an internal variable by its index.

        :param i_variable: Index of the internal variable to retrieve.
        :return: The internal variable object if found, otherwise None.
        """
        for o_internal_variable in self.des_funct_variables:
            if o_internal_variable.l_index == i_variable:
                return o_internal_variable
        return None  # Return None if the variable is not found

    def update_internal_variable(self, o_internal_variable_to_update):
        """
        Update an internal variable in the list of descriptive function variables.

        :param o_internal_variable_to_update: The internal variable object with updated values.
        """
        for i, o_internal_variable in enumerate(self.des_funct_variables):
            if o_internal_variable.l_index == o_internal_variable_to_update.l_index:
                self.des_funct_variables[i] = o_internal_variable_to_update
                return  # Exit after updating the variable

    @staticmethod
    def find_local_attractors(o_local_network, l_local_scenes=None):
        CustomText.print_simple_line()
        print("FIND ATTRACTORS FOR NETWORK:", o_local_network.index)

        if l_local_scenes is None:
            o_local_scene = LocalScene(index=1)
            o_local_scene.l_attractors = LocalNetwork.find_local_scene_attractors(o_local_network=o_local_network,
                                                                                  scene=None)
            o_local_network.l_local_scenes.append(o_local_scene)

            o_local_network.count_attractor = len(o_local_scene.l_attractors)

        else:
            v_scene_index = 1
            network_attractor_count = 0
            for scene in l_local_scenes:
                o_local_scene = LocalScene(v_scene_index, scene, o_local_network.l_var_exterm)
                s_scene = ''.join(scene)
                o_local_scene.l_attractors = LocalNetwork.find_local_scene_attractors(o_local_network=o_local_network,
                                                                                      scene=s_scene)
                o_local_network.l_local_scenes.append(o_local_scene)

                # update the scenes index
                v_scene_index += 1

                # update the attractors index
                network_attractor_count += len(o_local_scene.l_attractors)

            # Update the count attractor
            o_local_network.count_attractor = network_attractor_count

        return o_local_network

    @staticmethod
    def gen_boolean_formulation(o_local_network, n_transitions, l_attractors_clauses, scene):
        """
        Generate the boolean formulation for the given local network. This formulation includes:
        - CNF variable creation
        - Boolean expressions for each transition
        - Assignment values for permutations
        - Incorporation of attractor clauses

        :param o_local_network: The local network object for which to generate the boolean formulation.
        :param n_transitions: Number of transitions in the boolean formulation.
        :param l_attractors_clauses: List of clauses representing attractors.
        :param scene: Optional scene string used to assign 5_specific values to variables.
        :return: The complete boolean function as a Variable object.
        """
        # Create dictionary of CNF variables for each transition
        for variable in o_local_network.l_var_total:
            for transition_c in range(0, n_transitions):
                o_local_network.dic_var_cnf[f"{variable}_{transition_c}"] = Variable(f"{variable}_{transition_c}")

        cont_transition = 0
        boolean_function = Variable("0_0")

        for transition in range(1, n_transitions):
            cont_clause_global = 0
            boolean_expression_equivalence = Variable("0_0")

            for o_variable_model in o_local_network.des_funct_variables:
                cont_clause = 0
                boolean_expression_clause_global = Variable("0_0")

                for clause in o_variable_model.cnf_function:
                    boolean_expression_clause = Variable("0_0")
                    cont_term = 0

                    for term in clause:
                        term_aux = abs(int(term))
                        if cont_term == 0:
                            if str(term)[0] != "-":
                                boolean_expression_clause = o_local_network.dic_var_cnf[f"{term_aux}_{transition - 1}"]
                            else:
                                boolean_expression_clause = -o_local_network.dic_var_cnf[f"{term_aux}_{transition - 1}"]
                        else:
                            if str(term)[0] != "-":
                                boolean_expression_clause |= o_local_network.dic_var_cnf[f"{term_aux}_{transition - 1}"]
                            else:
                                boolean_expression_clause |= -o_local_network.dic_var_cnf[
                                    f"{term_aux}_{transition - 1}"]

                        cont_term += 1

                    if cont_clause == 0:
                        boolean_expression_clause_global = boolean_expression_clause
                    else:
                        boolean_expression_clause_global &= boolean_expression_clause

                    cont_clause += 1

                if cont_clause_global == 0:
                    boolean_expression_equivalence = o_local_network.dic_var_cnf[
                                                         f"{o_variable_model.index}_{transition}"] >> boolean_expression_clause_global
                    boolean_expression_equivalence &= boolean_expression_clause_global >> o_local_network.dic_var_cnf[
                        f"{o_variable_model.index}_{transition}"]
                else:
                    boolean_expression_equivalence &= o_local_network.dic_var_cnf[
                                                          f"{o_variable_model.index}_{transition}"] >> boolean_expression_clause_global
                    boolean_expression_equivalence &= boolean_expression_clause_global >> o_local_network.dic_var_cnf[
                        f"{o_variable_model.index}_{transition}"]

                if not o_variable_model.cnf_function:
                    print("ENTER ATYPICAL CASE!!!")
                    boolean_function &= (o_local_network.dic_var_cnf[f"{o_variable_model.index}_{transition}"] | -
                    o_local_network.dic_var_cnf[f"{o_variable_model.index}_{transition}"])

                cont_clause_global += 1

            if cont_transition == 0:
                boolean_function = boolean_expression_equivalence
            else:
                boolean_function &= boolean_expression_equivalence

            cont_transition += 1

        # Assign values for permutations if a scene is provided
        if scene is not None:
            cont_permutation = 0
            for element in o_local_network.l_var_exterm:
                for v_transition in range(0, n_transitions):
                    if scene[cont_permutation] == "0":
                        boolean_function &= -o_local_network.dic_var_cnf[f"{element}_{v_transition}"]
                    else:
                        boolean_function &= o_local_network.dic_var_cnf[f"{element}_{v_transition}"]

                cont_permutation += 1

        # Add attractors to the boolean function
        if l_attractors_clauses:
            boolean_function_of_attractors = Variable("0_0")
            cont_clause = 0

            for clause in l_attractors_clauses:
                bool_expr_clause_attractors = Variable("0_0")
                cont_term = 0

                for term in clause:
                    term_aux = abs(int(term))
                    if cont_term == 0:
                        if term[0] != "-":
                            bool_expr_clause_attractors = o_local_network.dic_var_cnf[f"{term_aux}_{n_transitions - 1}"]
                        else:
                            bool_expr_clause_attractors = -o_local_network.dic_var_cnf[
                                f"{term_aux}_{n_transitions - 1}"]
                    else:
                        if term[0] != "-":
                            bool_expr_clause_attractors &= o_local_network.dic_var_cnf[
                                f"{term_aux}_{n_transitions - 1}"]
                        else:
                            bool_expr_clause_attractors &= -o_local_network.dic_var_cnf[
                                f"{term_aux}_{n_transitions - 1}"]

                    cont_term += 1

                if cont_clause == 0:
                    boolean_function_of_attractors = -bool_expr_clause_attractors
                else:
                    boolean_function_of_attractors &= -bool_expr_clause_attractors

                cont_clause += 1

            boolean_function &= boolean_function_of_attractors

        # Ensure all variables are included in the boolean function
        for variable in o_local_network.l_var_total:
            boolean_function &= (
                    o_local_network.dic_var_cnf[f"{variable}_0"] | -o_local_network.dic_var_cnf[f"{variable}_0"])

        return boolean_function

    @staticmethod
    def find_local_scene_attractors(o_local_network, scene=None):
        def count_state_repeat(v_estate, path_candidate):
            # input type [[],[],...[]]
            number_of_times = 0
            for v_element in path_candidate:
                if v_element == v_estate:
                    number_of_times = number_of_times + 1
            return number_of_times

        CustomText.print_simple_line()
        print("Network:", o_local_network.index, " Local Scene:", scene)

        # First obligatory execution
        set_of_attractors = []
        v_num_transitions = 3
        l_attractors = []
        l_attractors_clauses = []

        # create boolean expression initial with 3 transitions
        v_boolean_formulation = o_local_network.gen_boolean_formulation(o_local_network=o_local_network,
                                                                        n_transitions=v_num_transitions,
                                                                        l_attractors_clauses=l_attractors_clauses,
                                                                        scene=scene)
        m_response_sat = []
        # Solve with SAT the boolean formulation
        o_solver = Minisat()
        o_solution = o_solver.solve(v_boolean_formulation)

        if o_solution.success:
            for j in range(0, v_num_transitions):
                m_response_sat.append([])
                for i in o_local_network.l_var_total:
                    m_response_sat[j].append(o_solution[o_local_network.dic_var_cnf[f'{i}_{j}']])

        m_aux_sat = []
        if len(m_response_sat) != 0:
            # TRANSFORM BOOLEAN TO MATRIZ BOOLEAN RESPONSE
            for j in range(0, v_num_transitions):
                matriz_aux_sat = []
                for i in range(0, o_local_network.num_var_total):
                    if m_response_sat[j][i]:
                        matriz_aux_sat.append("1")
                    else:
                        matriz_aux_sat.append("0")
                m_aux_sat.append(matriz_aux_sat)
            # m_resp_boolean = m_aux_sat
        m_resp_boolean = m_aux_sat
        # BLOCK ATTRACTORS
        # REPEAT CODE

        while len(m_resp_boolean) > 0:
            path_solution = []
            for path_transition in m_resp_boolean:
                path_solution.append(path_transition)

            # new list of state attractors
            l_news_estates_attractor = []
            # check attractors
            for v_state in path_solution:
                v_state_count = count_state_repeat(v_state, path_solution)
                if v_state_count > 1:
                    attractor_begin = path_solution.index(v_state) + 1
                    attractor_end = path_solution[attractor_begin:].index(v_state)
                    l_news_estates_attractor = path_solution[attractor_begin - 1:(attractor_begin + attractor_end)]
                    l_attractors = l_attractors + l_news_estates_attractor
                    # add attractors like list of list
                    set_of_attractors.append(l_news_estates_attractor)
                    break

            # print set_of_attractors
            if len(l_news_estates_attractor) == 0:
                # print ("duplicating")
                v_num_transitions = v_num_transitions * 2

            # TRANSFORM LIST OF ATTRACTORS TO CLAUSES
            for clause_attractor in l_attractors:
                clause_variable = []
                cont_variable = 0
                for estate_attractor in clause_attractor:
                    if estate_attractor == "0":
                        clause_variable.append("-" + str(o_local_network.l_var_total[cont_variable]))
                    else:
                        clause_variable.append(str(o_local_network.l_var_total[cont_variable]))
                    cont_variable = cont_variable + 1
                l_attractors_clauses.append(clause_variable)

            # print l_attractors_clauses
            # REPEAT CODE
            v_boolean_formulation = o_local_network.gen_boolean_formulation(o_local_network=o_local_network,
                                                                            n_transitions=v_num_transitions,
                                                                            l_attractors_clauses=l_attractors_clauses,
                                                                            scene=scene)
            m_response_sat = []
            o_solver = Minisat()
            o_solution = o_solver.solve(v_boolean_formulation)

            if o_solution.success:
                for j in range(0, v_num_transitions):
                    m_response_sat.append([])
                    for i in o_local_network.l_var_total:
                        m_response_sat[j].append(o_solution[o_local_network.dic_var_cnf[f'{i}_{j}']])
            # else:
            #     # print(" ")
            #     print("The expression cannot be satisfied")

            # BLOCK ATTRACTORS
            m_aux_sat = []
            if len(m_response_sat) != 0:
                # TRANSFORM BOOLEAN TO MATRIZ BOOLEAN RESPONSE
                for j in range(0, v_num_transitions):
                    matriz_aux_sat = []
                    for i in range(0, o_local_network.num_var_total):
                        if m_response_sat[j][i]:
                            matriz_aux_sat.append("1")
                        else:
                            matriz_aux_sat.append("0")
                    m_aux_sat.append(matriz_aux_sat)
                # m_resp_boolean = m_aux_sat
            m_resp_boolean = m_aux_sat
            # BLOCK ATTRACTORS
            # REPEAT CODE

        # Creating the objects of the attractor
        l_scene_attractors = []
        v_index = 1
        for o_attractor in set_of_attractors:
            l_local_states = []
            for o_state in o_attractor:
                o_local_state = LocalState(o_state)
                l_local_states.append(o_local_state)
            o_local_attractor = LocalAttractor(g_index=None,
                                               l_index=v_index,
                                               l_states=l_local_states,
                                               network_index=o_local_network.index,
                                               relation_index=o_local_network.l_var_exterm,
                                               local_scene=scene)

            l_scene_attractors.append(o_local_attractor)
            # update the attractors index locally
            v_index += 1

        print("end find attractors")
        return l_scene_attractors
