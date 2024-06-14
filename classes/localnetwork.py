# external imports
from satispy import Variable  # Library to resolve SAT
from satispy.solver import Minisat  # Library to resolve SAT

# internal imports
from classes.localscene import LocalScene, LocalAttractor, LocalState
from classes.utils.customtext import CustomText


class LocalNetwork:
    def __init__(self, num_local_network, l_var_intern):
        # basic properties
        self.index = num_local_network
        self.l_var_intern = l_var_intern

        # Processed properties
        self.des_funct_variables = []
        self.l_var_exterm = []
        self.l_var_total = []
        self.num_var_total = 0
        self.dic_var_cnf = {}

        self.l_input_signals = []
        self.l_output_signals = []

        # Calculated properties
        self.count_attractor = 1
        self.l_local_scenes = []

    def show(self):
        CustomText.make_sub_sub_title(f"Local Network: {self.index}")
        print('Internal Variables: ', self.l_var_intern)
        print('External Variables: ', self.l_var_exterm)
        print('Total Variables: ', self.l_var_total)
        # Description variables
        for o_internal_variable in self.des_funct_variables:
            o_internal_variable.show()

    def process_input_signals(self, l_input_signals):
        # Processing the input signals of local network
        for o_signal in l_input_signals:
            self.l_var_exterm.append(o_signal.index_variable)
        # add the local variables
        self.l_var_total.extend(self.l_var_intern.copy())
        # add the external variables from coupling signal
        self.l_var_total.extend(self.l_var_exterm.copy())
        # calculate the number of total variables
        self.num_var_total = len(self.l_var_total)

    def get_internal_variable(self, i_variable):
        for o_internal_variable in self.des_funct_variables:
            if o_internal_variable.l_index == i_variable:
                return o_internal_variable

    def update_internal_variable(self, o_internal_variable_to_update):
        for i, o_internal_variable in enumerate(self.des_funct_variables):
            if o_internal_variable.l_index == o_internal_variable_to_update.l_index:
                self.des_funct_variables[i] = o_internal_variable_to_update

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
        # create dictionary of cnf variables!!
        for variable in o_local_network.l_var_total:
            for transition_c in range(0, n_transitions):
                o_local_network.dic_var_cnf[str(variable) + "_" + str(transition_c)] = Variable(
                    str(variable) + "_" + str(transition_c))

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
                                boolean_expression_clause = o_local_network.dic_var_cnf[
                                    str(term_aux) + "_" + str(transition - 1)]
                            else:
                                boolean_expression_clause = -o_local_network.dic_var_cnf[
                                    str(term_aux) + "_" + str(transition - 1)]
                        else:
                            if str(term)[0] != "-":
                                boolean_expression_clause = o_local_network.dic_var_cnf[str(term_aux) + "_" + str(
                                    transition - 1)] | boolean_expression_clause
                            else:
                                boolean_expression_clause = -o_local_network.dic_var_cnf[
                                    str(term_aux) + "_" + str(transition - 1)] | boolean_expression_clause
                        cont_term = cont_term + 1
                    if cont_clause == 0:
                        boolean_expression_clause_global = boolean_expression_clause
                    else:
                        boolean_expression_clause_global = boolean_expression_clause_global & boolean_expression_clause
                    cont_clause = cont_clause + 1
                if cont_clause_global == 0:
                    boolean_expression_equivalence = o_local_network.dic_var_cnf[
                                                         str(o_variable_model.index) + "_" + str(
                                                             transition)] >> boolean_expression_clause_global
                    boolean_expression_equivalence = boolean_expression_equivalence & (
                            boolean_expression_clause_global >> o_local_network.dic_var_cnf[
                        str(o_variable_model.index) + "_" + str(transition)])
                else:
                    boolean_expression_equivalence = boolean_expression_equivalence & (o_local_network.dic_var_cnf[
                                                                                           str(o_variable_model.index) + "_" + str(
                                                                                               transition)] >> boolean_expression_clause_global)
                    boolean_expression_equivalence = boolean_expression_equivalence & (
                            boolean_expression_clause_global >> o_local_network.dic_var_cnf[
                        str(o_variable_model.index) + "_" + str(transition)])
                if not o_variable_model.cnf_function:
                    print("ENTER ATYPICAL CASE!!!")
                    boolean_function = boolean_function & (
                            o_local_network.dic_var_cnf[str(o_variable_model.index) + "_" + str(transition)] | -
                    o_local_network.dic_var_cnf[str(o_variable_model.index) + "_" + str(transition)])
                cont_clause_global = cont_clause_global + 1
            if cont_transition == 0:
                boolean_function = boolean_expression_equivalence
            else:
                boolean_function = boolean_function & boolean_expression_equivalence
            # validate blank gens
            cont_transition = cont_transition + 1

        # ASSIGN VALUES FOR PERMUTATIONS
        if scene is not None:
            cont_permutation = 0
            for element in o_local_network.l_var_exterm:
                # print oRDD.list_of_v_exterm
                for v_transition in range(0, n_transitions):
                    # print l_signal_coupling[cont_permutation]
                    if scene[cont_permutation] == "0":
                        boolean_function = boolean_function & -o_local_network.dic_var_cnf[
                            str(element) + "_" + str(v_transition)]
                        # print (str(element) +"_"+ str(v_transition))
                    else:
                        boolean_function = boolean_function & o_local_network.dic_var_cnf[
                            str(element) + "_" + str(v_transition)]
                        # print (str(element) +"_"+ str(v_transition))
                cont_permutation = cont_permutation + 1

        # add attractors to boolean function
        if len(l_attractors_clauses) > 0:
            boolean_function_of_attractors = Variable("0_0")
            cont_clause = 0
            for clause in l_attractors_clauses:
                bool_expr_clause_attractors = Variable("0_0")
                cont_term = 0
                for term in clause:
                    term_aux = abs(int(term))
                    if cont_term == 0:
                        if term[0] != "-":
                            bool_expr_clause_attractors = o_local_network.dic_var_cnf[
                                str(term_aux) + "_" + str(n_transitions - 1)]
                        else:
                            bool_expr_clause_attractors = -o_local_network.dic_var_cnf[
                                str(term_aux) + "_" + str(n_transitions - 1)]
                    else:
                        if term[0] != "-":
                            bool_expr_clause_attractors = bool_expr_clause_attractors & \
                                                          o_local_network.dic_var_cnf[
                                                              str(term_aux) + "_" + str(
                                                                  n_transitions - 1)]
                        else:
                            bool_expr_clause_attractors = bool_expr_clause_attractors & - \
                                o_local_network.dic_var_cnf[str(term_aux) + "_" + str(n_transitions - 1)]
                    cont_term = cont_term + 1
                if cont_clause == 0:
                    boolean_function_of_attractors = -bool_expr_clause_attractors
                else:
                    boolean_function_of_attractors = boolean_function_of_attractors & - bool_expr_clause_attractors
                cont_clause = cont_clause + 1
            boolean_function = boolean_function & boolean_function_of_attractors

        # Add all the variables of position 0 to the boolean function
        for variable in o_local_network.l_var_total:
            boolean_function = boolean_function & (o_local_network.dic_var_cnf[str(variable) + "_0"] |
                                                   - o_local_network.dic_var_cnf[str(variable) + "_0"])
        # print(boolean_function)
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
