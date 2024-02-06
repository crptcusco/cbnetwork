# external imports
import itertools
import random                                           # generate random numbers
import networkx as nx                                   # generate networks
from itertools import product                           # generate the permutations te result in tuples
from random import randint                              # generate random numbers integers
import random
from satispy import Variable                            # Library to resolve SAT
from satispy.solver import Minisat                      # Library to resolve SAT
import re                                               # analysis of regular expressions
import operator                                         # unary operator management
import heapq

from string import ascii_lowercase, ascii_uppercase     # import the list of uppercase and lowercase letters
from itertools import product                           # generate combinations of numbers
from collections import namedtuple                      # structures like trees

# external imports
import os
import time
import pandas as pd
from memory_profiler import profile


class Node:
    def __init__(self, index, weight):
        self.index = index
        self.weight = weight

    def __lt__(self, other):
        # Define the comparison to sort nodes in the heap based on the weight
        return self.weight < other.weight


class CustomHeap:
    def __init__(self):
        self.heap = []

    def add_node(self, node):
        heapq.heappush(self.heap, node)

    def remove_node(self):
        if self.heap:
            return heapq.heappop(self.heap)
        else:
            return None

    def get_size(self):
        return len(self.heap)

    def get_indexes(self):
        indexes = []
        for node in self.heap:
            indexes.append(node.index)
        return indexes

    def update_node(self, index, new_weight):
        # Find the node with the specified index
        for i, node in enumerate(self.heap):
            if node.index == index:
                # Update the weight of the node
                node.weight = new_weight

                # Reorganize the heap to maintain the heap property
                heapq.heapify(self.heap)


class CustomText:
    @staticmethod
    def print_duplex_line():
        print("=================================================")

    @staticmethod
    def print_simple_line():
        print("-------------------------------------------------")

    @staticmethod
    def print_message(message, show):
        if show:
            print(message)

    @classmethod
    def print_stars(cls):
        print("*************************************************")

    @classmethod
    def print_dollars(cls):
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")


class DirectedEdge:
    def __init__(self, index_variable_signal, input_local_network, output_local_network, l_output_variables,
                 coupling_function):
        self.index_variable = index_variable_signal
        self.input_local_network = input_local_network
        self.output_local_network = output_local_network
        self.l_output_variables = l_output_variables
        self.coupling_function = coupling_function

        # Calculated properties
        # True table for signal with the output variables
        self.true_table = self.process_true_table()
        # Dictionary for kind status of the signal
        self.d_kind_signal = {1: "RESTRICTED",
                              2: "NOT COMPUTE",
                              3: "STABLE",
                              4: "NOT STABLE"}
        # Defined the initial kind for every coupling signal
        self.kind_signal = 2
        # Dictionary for group the attractors by his output signal value
        self.d_out_value_to_attractor = {1: [], 0: []}
        # List of the compatible pair attractors
        self.d_comp_pairs_attractors_by_value = {0: [], 1: []}

    def show(self):
        CustomText.print_simple_line()
        print("Edge:", self.output_local_network, "->", self.input_local_network,
              ", Index:", self.index_variable)
        print("Variables:", self.l_output_variables, ", Coupling Function:", self.coupling_function)
        print("Truth table:", self.true_table)
        print("Kind signal:", self.kind_signal,
              "-", self.d_kind_signal[self.kind_signal])

    def process_true_table(self):
        r_true_table = {}
        # print("Generating the True Table")
        # First we must understand the coupling signal
        # we will use regular expressions to recognize the boolean formula

        # TOKENIZATION
        # Regular expression matching optional whitespace followed by a token
        # (if group 1 matches) or an error (if group 2 matches).
        TOKEN_RE = re.compile(r'\s*(?:([A-Za-z01()~∧∨→↔])|(\S))')

        # Special token indicating the end of the input string.
        TOKEN_END = '<end of input>'

        def tokenize(s):
            """Generate tokens from the string s, followed by TOKEN_END."""
            for match in TOKEN_RE.finditer(s):
                token, error = match.groups()
                if token:
                    yield token
                else:
                    raise SyntaxError("Unexpected character {!r}".format(error))
            yield TOKEN_END

        # PARSING
        Constant = namedtuple('Constant', 'value')
        Variable = namedtuple('Variable', 'name')
        UnaryOp = namedtuple('UnaryOp', 'op operand')
        BinaryOp = namedtuple('BinaryOp', 'left op right')

        # Tokens representing Boolean constants (0=False, 1=True).
        CONSTANTS = '01'

        # Tokens representing variables.
        VARIABLES = set(ascii_lowercase) | set(ascii_uppercase)

        # Map from unary operator to function implementing it.
        UNARY_OPERATORS = {
            '~': operator.not_,
        }

        # Map from binary operator to function implementing it.
        BINARY_OPERATORS = {
            '∧': operator.and_,
            '∨': operator.or_,
            '→': lambda a, b: not a or b,
            '↔': operator.eq,
        }

        def parse(s):
            """Parse s as a Boolean expression and return the parse tree."""
            tokens = tokenize(s)  # Stream of tokens.
            token = next(tokens)  # The current token.

            def error(expected):
                # Current token failed to match, so raise syntax error.
                raise SyntaxError("Expected {} but found {!r}"
                                  .format(expected, token))

            def match(valid_tokens):
                # If the current token is found in valid_tokens, consume it
                # and return True. Otherwise, return False.
                nonlocal token
                if token in valid_tokens:
                    token = next(tokens)
                    return True
                else:
                    return False

            def term():
                # Parse a <Term> starting at the current token.
                t = token
                if match(VARIABLES):
                    return Variable(name=t)
                elif match(CONSTANTS):
                    return Constant(value=(t == '1'))
                elif match('('):
                    tree = disjunction()
                    if match(')'):
                        return tree
                    else:
                        error("')'")
                else:
                    error("term")

            def unary_expr():
                # Parse a <UnaryExpr> starting at the current token.
                t = token
                if match('~'):
                    operand = unary_expr()
                    return UnaryOp(op=UNARY_OPERATORS[t], operand=operand)
                else:
                    return term()

            def binary_expr(parse_left, valid_operators, parse_right):
                # Parse a binary expression starting at the current token.
                # Call parse_left to parse the left operand; the operator must
                # be found in valid_operators; call parse_right to parse the
                # right operand.
                left = parse_left()
                t = token
                if match(valid_operators):
                    right = parse_right()
                    return BinaryOp(left=left, op=BINARY_OPERATORS[t], right=right)
                else:
                    return left

            def implication():
                # Parse an <Implication> starting at the current token.
                return binary_expr(unary_expr, '→↔', implication)

            def conjunction():
                # Parse a <Conjunction> starting at the current token.
                return binary_expr(implication, '∧', conjunction)

            def disjunction():
                # Parse a <Disjunction> starting at the current token.
                return binary_expr(conjunction, '∨', disjunction)

            tree = disjunction()
            if token != TOKEN_END:
                error("end of input")
            return tree

        def evaluate(tree, env):
            """Evaluate the expression in the parse tree in the context of an
            environment mapping variable names to their values.
            """
            if isinstance(tree, Constant):
                return tree.value
            elif isinstance(tree, Variable):
                return env[tree.name]
            elif isinstance(tree, UnaryOp):
                return tree.op(evaluate(tree.operand, env))
            elif isinstance(tree, BinaryOp):
                return tree.op(evaluate(tree.left, env), evaluate(tree.right, env))
            else:
                raise TypeError("Expected tree, found {!r}".format(type(tree)))

        # we have to create a dictionary for each variable in the output set
        l_abecedario = list(ascii_uppercase)

        dict_aux_var_saida = {}
        cont_aux_abecedario = 0
        for variable_saida in self.l_output_variables:
            dict_aux_var_saida[" " + str(variable_saida) + " "] = l_abecedario[cont_aux_abecedario]
            cont_aux_abecedario = cont_aux_abecedario + 1

        # generate combinations of the output signal
        l_permutations = []
        for v_permutacion in product([True, False], repeat=len(self.l_output_variables)):
            l_permutations.append(v_permutacion)

        # process each of the permutations we simply have to evaluate and solve
        for c_permutation in l_permutations:
            aux_dictionary = dict(zip(dict_aux_var_saida.values(), c_permutation))
            aux_acoplament_function = self.coupling_function
            for aux_element in dict_aux_var_saida.keys():
                aux_acoplament_function = aux_acoplament_function.replace(str(aux_element),
                                                                          str(dict_aux_var_saida[aux_element]))
            # print("========= Signal =========")
            # print(aux_acoplament_function)
            # print(dict_aux_var_saida)
            # print(aux_dictionary)
            # print("========= End Signal =========")
            # Creating the key of the truth table
            aux_key = ''
            for v_literal in c_permutation:
                if v_literal:
                    aux_key = aux_key + "1"
                else:
                    aux_key = aux_key + "0"
            if evaluate(parse(aux_acoplament_function), aux_dictionary):
                r_true_table[aux_key] = "1"
            else:
                r_true_table[aux_key] = "0"

        # print the true table
        # print(r_true_table)
        # sys.exit()

        return r_true_table

    def show_dict_v_output_signal_attractor(self):
        for signal_value, l_attractors in self.d_out_value_to_attractor.items():
            print(signal_value, "-", l_attractors)

    def show_v_output_signal_attractor(self):
        for signal_value, l_attractors in self.d_out_value_to_attractor.items():
            print("Output signal Value -", signal_value, "- Attractors:")
            for o_attractor in l_attractors:
                o_attractor.show()

    # def show_d_comp_pairs_attractors_by_value(self, value):
    #     self.d_comp_pairs_attractors_by_value()
    #


class GlobalScene:
    def __init__(self, index, l_signal_indexes, l_values_signals):
        self.index = index
        self.l_signal_indexes = l_signal_indexes
        self.l_values_signals = l_values_signals

    def show(self):
        print("-------------------------------")
        print("Index Global Scene:", self.index)
        print("Indexes Directed Edges:", self.l_signal_indexes)
        print("Directed Edges Values:", self.l_values_signals)


# VARIABLE MODEL ONLY HAVE VARIABLE_NAME, CNF FUNCTION
class InternalVariable:
    index = 0
    cnf_function = []

    def __init__(self, index, cnf_function):
        self.index = int(index)
        self.cnf_function = cnf_function

    def show(self):
        print("Variable Index: " + str(self.index) + " -> CNF :" + str(self.cnf_function))


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
        self.l_local_scenes = []

    def show(self):
        print("--------------------------------------------")
        print('Local Network:', self.index)
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
            if o_internal_variable.index == i_variable:
                return o_internal_variable

    def update_internal_variable(self, o_internal_variable_to_update):
        for i, o_internal_variable in enumerate(self.des_funct_variables):
            if o_internal_variable.index == o_internal_variable_to_update.index:
                self.des_funct_variables[i] = o_internal_variable_to_update

    @staticmethod
    def find_local_attractors(o_local_network, l_local_scenes=None):
        CustomText.print_simple_line()
        print("FIND ATTRACTORS FOR NETWORK:", o_local_network.index)
        if l_local_scenes is None:
            o_local_scene = LocalScene(index=1)
            o_local_scene.l_attractors = LocalNetwork.find_local_scene_attractors(o_local_network, scene=None)
            o_local_network.l_local_scenes.append(o_local_scene)
        else:
            v_cont_index = 1
            for scene in l_local_scenes:
                o_local_scene = LocalScene(v_cont_index, scene, o_local_network.l_var_exterm)
                s_scene = ''.join(scene)
                o_local_scene.l_attractors = LocalNetwork.find_local_scene_attractors(o_local_network, s_scene)
                o_local_network.l_local_scenes.append(o_local_scene)
                v_cont_index = v_cont_index + 1
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
        # print("Begin to find attractors...")
        # create boolean expression initial with "n" transitions
        set_of_attractors = []
        v_num_transitions = 3
        l_attractors = []
        l_attractors_clauses = []

        # REPEAT CODE
        v_bool_function = o_local_network.gen_boolean_formulation(o_local_network, v_num_transitions,
                                                                  l_attractors_clauses, scene)
        m_response_sat = []
        o_solver = Minisat()
        o_solution = o_solver.solve(v_bool_function)

        if o_solution.success:
            for j in range(0, v_num_transitions):
                m_response_sat.append([])
                for i in o_local_network.l_var_total:
                    m_response_sat[j].append(o_solution[o_local_network.dic_var_cnf[f'{i}_{j}']])
        # attractor_begin

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

        while len(m_resp_boolean) > 0:
            # print ("path")
            # print (m_resp_boolean)
            # print ("path")
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
            v_bool_function = o_local_network.gen_boolean_formulation(o_local_network, v_num_transitions,
                                                                      l_attractors_clauses, scene)
            m_response_sat = []
            o_solver = Minisat()
            o_solution = o_solver.solve(v_bool_function)

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
        res = []
        v_index = 1
        for o_attractor in set_of_attractors:
            l_local_states = []
            for o_state in o_attractor:
                o_local_state = LocalState(o_state)
                l_local_states.append(o_local_state)
            o_local_attractor = LocalAttractor(v_index, l_local_states, o_local_network.index,
                                               o_local_network.l_var_exterm, scene)
            res.append(o_local_attractor)
            v_index = v_index + 1

        print("end find attractors")
        return res


class LocalScene:
    def __init__(self, index, l_values=None, l_index_signals=None):
        self.index = index
        self.l_values = l_values
        self.l_index_signals = l_index_signals
        # Calculated properties
        self.l_attractors = []


class LocalAttractor:
    def __init__(self, index, l_states, network_index, relation_index=None, local_scene=None):
        # base properties
        self.index = index
        self.l_states = l_states
        # extended properties
        self.network_index = network_index
        self.relation_index = relation_index
        self.local_scene = local_scene

    def show(self):
        print("Network Index:", self.network_index, ", Input Signal Index:", self.relation_index,
              ", Scene:", self.local_scene, ", Attractor Index:", self.index, ", States:", end="")
        for o_state in self.l_states:
            print(end='[')
            for variable in o_state.l_variable_values:
                print(variable, end=",")
            print(end=']')
        print()

    def show_short(self):
        print("Net. Index:", self.network_index, ", Attrac. Index:", self.index, ", States:", end="")
        for o_state in self.l_states:
            print(end='[')
            for variable in o_state.l_variable_values:
                print(variable, end=",")
            print(end=']')
        print()


class LocalState:
    def __init__(self, l_variable_values):
        self.l_variable_values = l_variable_values


class CBN:
    def __init__(self, l_local_networks, l_directed_edges):
        # basic attributes
        self.l_local_networks = l_local_networks
        self.l_directed_edges = l_directed_edges

        # calculated attributes
        self.l_global_scenes = []
        self.l_attractor_fields = []

    # FUNCTIONS
    @staticmethod
    def generate_cbn_topology(n_nodes, v_topology=1):
        # Generate a directed graph begin in 1
        G = nx.DiGraph()
        # classical topologies
        # complete_graph
        if v_topology == 1:
            G = nx.complete_graph(n_nodes, nx.DiGraph())
        # binomial_tree
        elif v_topology == 2:
            G = nx.binomial_tree(n_nodes, nx.DiGraph())
        # cycle_graph
        elif v_topology == 3:
            G = nx.cycle_graph(n_nodes, nx.DiGraph())
        # path_graph
        elif v_topology == 4:
            G = nx.path_graph(n_nodes, nx.DiGraph())
        # aleatory topologies
        # gn_graph
        elif v_topology == 5:
            G = nx.gn_graph(n_nodes)
        elif v_topology == 6:
            G = nx.gnc_graph(n_nodes)
        # linear_graph
        elif v_topology == 7:
            G = nx.DiGraph()
            G.add_nodes_from(range(1, n_nodes + 1))
            for i in range(1, n_nodes):
                G.add_edge(i, i + 1)
        else:
            G = nx.complete_graph(n_nodes, nx.DiGraph())

        # Renaming the label of the nodes for beginning in 1
        mapping = {node: node + 1 for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        return list(G.edges)

    @staticmethod
    def generate_local_networks_indexes_variables(n_local_networks, n_var_network):
        l_local_networks = []
        v_cont_var = 1
        for v_num_network in range(1, n_local_networks + 1):
            # generate the variables of the networks
            l_var_intern = list(range(v_cont_var, v_cont_var + n_var_network))
            # create the Local Network object
            o_local_network = LocalNetwork(v_num_network, l_var_intern)
            # add the local network object to the list
            l_local_networks.append(o_local_network)
            # update the index of the variables
            v_cont_var = v_cont_var + n_var_network
        return l_local_networks

    @staticmethod
    def generate_directed_edges(i_last_variable, l_local_networks, l_relations, n_output_variables=2):
        l_directed_edges = []
        i_directed_edge = i_last_variable + 1

        # aux1_l_local_networks = []
        for o_local_network in l_local_networks:
            l_local_networks_co = []
            for t_relation in l_relations:
                if t_relation[1] == o_local_network.index:
                    o_local_network_aux = next(filter(lambda x: x.index == t_relation[0], l_local_networks), None)
                    l_local_networks_co.append(o_local_network_aux)

            for o_local_network_co in l_local_networks_co:
                l_output_variables = random.sample(o_local_network_co.l_var_intern, n_output_variables)
                if n_output_variables == 1:
                    coupling_function = l_output_variables[0]
                else:
                    coupling_function = " " + " ∨ ".join(list(map(str, l_output_variables))) + " "
                # generate the directed-edge object
                o_directed_edge = DirectedEdge(i_directed_edge, o_local_network.index, o_local_network_co.index,
                                               l_output_variables, coupling_function)
                l_directed_edges.append(o_directed_edge)
                i_directed_edge = i_directed_edge + 1
        #     aux1_l_local_networks.append(l_var_intern)
        # l_local_networks = aux1_l_local_networks.copy()

        return l_directed_edges

    @staticmethod
    def find_input_edges_by_network_index(index, l_directed_edges):
        res = []
        for o_directed_edge in l_directed_edges:
            if o_directed_edge.input_local_network == index:
                res.append(o_directed_edge)
        return res

    @staticmethod
    def find_output_edges_by_network_index(index, l_directed_edges):
        res = []
        for o_directed_edge in l_directed_edges:
            if o_directed_edge.output_local_network == index:
                res.append(o_directed_edge)
        return res

    @staticmethod
    def generate_local_networks_variables_dynamic(l_local_networks, l_directed_edges, n_input_variables=2):
        # GENERATE THE DYNAMICS OF EACH LOCAL NETWORK
        number_max_of_clauses = 2
        number_max_of_literals = 3
        # we generate an auxiliary list to add the coupling signals
        l_local_networks_updated = []
        for o_local_network in l_local_networks:
            # Create a list of all RDDAs variables
            l_aux_variables = []
            # Add the variables of the coupling signals
            l_input_signals = CBN.find_input_edges_by_network_index(o_local_network.index, l_directed_edges)
            for o_signal in l_input_signals:
                l_aux_variables.append(o_signal.index_variable)
            # add local variables
            l_aux_variables.extend(o_local_network.l_var_intern)

            # generate a dictionary for save the dynamic for every variable
            d_literals_variables = {}

            # generate the function description of the variables
            des_funct_variables = []
            # generate clauses
            for i_local_variable in o_local_network.l_var_intern:
                l_clauses_node = []
                for v_clause in range(0, randint(1, number_max_of_clauses)):
                    v_num_variable = randint(1, number_max_of_literals)
                    # randomly select from the signal variables
                    l_literals_variables = random.sample(l_aux_variables, v_num_variable)
                    l_clauses_node.append(l_literals_variables)

                # generate the Internal Variable Object with his index and his list of clauses
                o_internal_variable = InternalVariable(i_local_variable, l_clauses_node)
                # adding the description in functions of every variable
                des_funct_variables.append(o_internal_variable)

            # add the CNF variable dynamic in list of Satispy variables format
            o_local_network.des_funct_variables = des_funct_variables.copy()

            # adding the local network to a list of local networks
            l_local_networks_updated.append(o_local_network)
            print("Local network created :", o_local_network.index)
            CustomText.print_simple_line()

        # actualized the list of local networks
        return l_local_networks_updated

    @staticmethod
    def generate_local_networks_dynamic_from_template(l_local_networks, l_directed_edges, n_input_variables,
                                                      o_local_network_template):
        pass

    @staticmethod
    def generate_cbn(n_local_networks, n_var_network, v_topology, n_output_variables=2, n_input_variables=2):
        """
         Generates an instance of a CBN.

         Args:
             n_local_networks (int): The total number of local networks
             n_var_network (int): The total number of variables by local network
             v_topology (int): The topology of the global network
             n_output_variables (int): The number of output variables
             n_input_variables (int): The number of input variables
             o_local_network_template (object): A template for all the local networks

         Returns:
             CBN: The generated CBN object
         """

        # generate the local networks with the indexes and variables (without relations or dynamics)
        l_local_networks = CBN.generate_local_networks_indexes_variables(n_local_networks, n_var_network)

        # generate the CBN topology
        l_relations = CBN.generate_cbn_topology(n_local_networks, v_topology)

        # search the last variable from the local network variables
        i_last_variable = l_local_networks[-1].l_var_intern[-1]

        # generate the directed edges given the last variable generated
        l_directed_edges = CBN.generate_directed_edges(i_last_variable=i_last_variable,
                                                       l_local_networks=l_local_networks,
                                                       l_relations=l_relations,
                                                       n_output_variables=n_output_variables)

        # Process the coupling signals for every local network
        for o_local_network in l_local_networks:
            # find the signals for every local network
            l_input_signals = CBN.find_input_edges_by_network_index(o_local_network.index, l_directed_edges)
            o_local_network.process_input_signals(l_input_signals)

        # generate the local network dynamic
        l_local_networks = CBN.generate_local_networks_variables_dynamic(l_local_networks=l_local_networks,
                                                                         l_directed_edges=l_directed_edges,
                                                                         n_input_variables=n_input_variables)

        # create the cbn object
        o_cbn = CBN(l_local_networks, l_directed_edges)
        return o_cbn

    def process_output_signals(self):
        # update output signals for every local network
        for o_local_network in self.l_local_networks:
            for t_relation in self.l_directed_edges:
                if o_local_network.index == t_relation[1]:
                    o_local_network.l_output_signals.append(t_relation)
                    print(t_relation)

    def update_network_by_index(self, o_local_network_update):
        for i, o_local_network in enumerate(self.l_local_networks):
            if o_local_network.index == o_local_network_update.index:
                self.l_local_networks[i] = o_local_network_update
                print("Local Network updated")
                return True
        print("ERROR:", "Local Network not found")
        return False

    def generate_global_scenes(self):
        CustomText.print_duplex_line()
        print("GENERATE GLOBAL SCENES")

        # get the index for every directed_edge
        l_global_signal_indexes = []
        for o_directed_edge in self.l_directed_edges:
            l_global_signal_indexes.append(o_directed_edge.index_variable)

        # generate the global scenes using all the combinations
        l_global_scenes_values = list(product(list('01'), repeat=len(self.l_directed_edges)))

        cont_index_scene = 1
        for global_scene_values in l_global_scenes_values:
            o_global_scene = GlobalScene(cont_index_scene, l_global_signal_indexes, global_scene_values)
            self.l_global_scenes.append(o_global_scene)
            cont_index_scene = cont_index_scene + 1

        CustomText.print_simple_line()
        print("Global Scenes generated")

    def find_local_attractors_optimized(self):
        CustomText.print_duplex_line()
        print("FIND ATTRACTORS USING OPTIMIZED METHOD")

        # create an empty heap to organize the local networks by weight
        o_custom_heap = CustomHeap()

        # calculate the initial weights for every local network anda safe in the node of the heap
        for o_local_network in self.l_local_networks:
            weight = 0
            for o_directed_edge in self.l_directed_edges:
                if o_directed_edge.input_local_network == o_local_network.index:
                    # In the beginning all the kind or relations are "not computed" with index 2
                    weight = weight + o_directed_edge.kind_signal
            # create the node of the heap
            o_node = Node(o_local_network.index, weight)
            # add node to the heap with computed weight
            o_custom_heap.add_node(o_node)

        # generate the initial heap
        initial_heap = o_custom_heap.get_indexes()
        # print(initial_heap)

        # find the node in the top  of the heap
        lowest_weight_node = CustomHeap.remove_node(o_custom_heap)
        # find the local network information
        o_local_network = self.get_network_by_index(lowest_weight_node.index)
        # generate the local scenarios
        l_local_scenes = None
        if len(o_local_network.l_var_exterm) != 0:
            l_local_scenes = list(product(list('01'), repeat=len(o_local_network.l_var_exterm)))

        # calculate the attractors for the node in the top of the  heap
        o_local_network = LocalNetwork.find_local_attractors(o_local_network, l_local_scenes)
        # update the network in the CBN
        self.update_network_by_index(o_local_network)

        # validate if the output variables by attractor send a fixed value and update kind signals
        l_directed_edges = CBN.find_output_edges_by_network_index(o_local_network.index, self.l_directed_edges)
        # print("Local network:", l_var_intern.index)
        for o_output_signal in l_directed_edges:
            # print("Index variable output signal:", o_output_signal.index_variable_signal)
            # print("Output variables:", o_output_signal.l_output_variables)
            # print(str(o_output_signal.true_table))
            l_signals_for_output = []
            for o_local_scene in o_local_network.l_local_scenes:
                # print("Scene: ", str(o_local_scene.l_values))
                l_signals_in_local_scene = []
                for o_attractor in o_local_scene.l_attractors:
                    # print("ATTRACTOR")
                    l_signals_in_attractor = []
                    for o_state in o_attractor.l_states:
                        # print("STATE")
                        # print(l_var_intern.l_var_total)
                        # print(l_var_intern.l_var_intern)
                        # print(o_state.l_variable_values)
                        # # select the values of the output variables
                        true_table_index = ""
                        for v_output_variable in o_output_signal.l_output_variables:
                            # print("Variables list:", l_var_intern.l_var_total)
                            # print("Output variables list:", o_output_signal.l_output_variables)
                            # print("Output variable:", v_output_variable)
                            pos = o_local_network.l_var_total.index(v_output_variable)
                            value = o_state.l_variable_values[pos]
                            true_table_index = true_table_index + str(value)
                        # print(o_output_signal.l_output_variables)
                        # print(true_table_index)
                        output_value_state = o_output_signal.true_table[true_table_index]
                        # print("Output value :", output_value_state)
                        l_signals_in_attractor.append(output_value_state)
                    if len(set(l_signals_in_attractor)) == 1:
                        l_signals_in_local_scene.append(l_signals_in_attractor[0])
                        # print("the attractor signal value is stable")

                        # add the attractor to the dictionary of output value -> attractors
                        if l_signals_in_attractor[0] == '0':
                            o_output_signal.d_out_value_to_attractor[0].append(o_attractor)
                        elif l_signals_in_attractor[0] == '1':
                            o_output_signal.d_out_value_to_attractor[1].append(o_attractor)
                    # else:
                    #     print("the attractor signal is not stable")
                if len(set(l_signals_in_local_scene)) == 1:
                    l_signals_for_output.append(l_signals_in_local_scene[0])
                    # print("the scene signal is restricted")
                else:
                    if len(set(l_signals_in_local_scene)) == 2:
                        l_signals_for_output.extend(l_signals_in_local_scene)
                        # print("the scene signal value is stable")
                    # else:
                    #     print("warning:", "the scene signal is not stable")
            if len(set(l_signals_for_output)) == 1:
                o_output_signal.kind_signal = 1
                print("the output signal is restricted")
            elif len(set(l_signals_for_output)) == 2:
                o_output_signal.kind_signal = 3
                print("the output signal is stable")
            else:
                o_output_signal.kind_signal = 4
                print("error:", "the scene signal is not stable. This CBN dont have stable Attractor Fields")

        # # # print all the kinds of the signals
        # CustomText.print_simple_line()
        # print("Resume")
        # print("Network:", l_var_intern.index)
        # for o_directed_edge in self.l_directed_edges:
        #     print(o_directed_edge.index_variable, ":", o_directed_edge.kind_signal)

        # Update the weights of the nodes
        # Add the output network to the list of modified networks
        l_modified_edges = CBN.find_input_edges_by_network_index(o_local_network.index, self.l_directed_edges)
        for o_edge in l_modified_edges:
            modified_network_index = o_edge.output_local_network
            # print("Network", modified_network_index)
            # print("Relation:", o_edge.input_local_network, "->", o_edge.output_local_network)
            weight = 0
            l_edges = CBN.find_input_edges_by_network_index(o_edge.output_local_network, self.l_directed_edges)
            for o_updated_edge in l_edges:
                weight = weight + o_updated_edge.kind_signal
            # print("New weight:", weight)
            o_custom_heap.update_node(o_edge.output_local_network, weight)

        # # compare the initial heap with the update heap
        # print("INITIAL HEAP")
        # print(initial_heap)
        # print("UPDATE HEAP")
        # print(o_custom_heap.get_indexes())

        # Verify if the heap has at least two elements
        while o_custom_heap.get_size() > 0:
            # find the node on the top of the heap
            lowest_weight_node = CustomHeap.remove_node(o_custom_heap)
            # Find Local Network
            o_local_network = self.get_network_by_index(lowest_weight_node.index)

            l_local_scenes = None
            if len(o_local_network.l_var_exterm) != 0:
                l_local_scenes = list(product(list('01'), repeat=len(o_local_network.l_var_exterm)))

            # Find attractors with the minimum weight
            LocalNetwork.find_local_attractors(o_local_network, l_local_scenes)
            # print("Local Network:", lowest_weight_node.index, "Weight:", lowest_weight_node.weight)

            # COPIED CODE !!!
            # # Update kind signals
            # validate if the output variables by attractor send a fixed value
            l_directed_edges = CBN.find_output_edges_by_network_index(o_local_network.index,
                                                                      self.l_directed_edges)
            # print("Local network:", l_var_intern.index)
            for o_output_signal in l_directed_edges:
                # print("Index variable output signal:", o_output_signal.index_variable)
                # print("Output variables:", o_output_signal.l_output_variables)
                # print(str(o_output_signal.true_table))
                l_signals_for_output = []
                for o_local_scene in o_local_network.l_local_scenes:
                    # print("Scene: ", str(o_local_scene.l_values))
                    l_signals_in_local_scene = []
                    for o_attractor in o_local_scene.l_attractors:
                        # print("ATTRACTOR")
                        l_signals_in_attractor = []
                        for o_state in o_attractor.l_states:
                            # print("STATE")
                            # print(l_var_intern.l_var_total)
                            # print(l_var_intern.l_var_intern)
                            # print(o_state.l_variable_values)
                            # select the values of the output variables
                            true_table_index = ""
                            for v_output_variable in o_output_signal.l_output_variables:
                                # print("Variables list:", l_var_intern.l_var_total)
                                # print("Output variables list:", o_output_signal.l_output_variables)
                                # print("Output variable:", v_output_variable)
                                pos = o_local_network.l_var_total.index(v_output_variable)
                                value = o_state.l_variable_values[pos]
                                true_table_index = true_table_index + str(value)
                            # print(o_output_signal.l_output_variables)
                            # print(true_table_index)
                            output_value_state = o_output_signal.true_table[true_table_index]
                            # print("Output value :", output_value_state)
                            l_signals_in_attractor.append(output_value_state)
                        if len(set(l_signals_in_attractor)) == 1:
                            l_signals_in_local_scene.append(l_signals_in_attractor[0])
                            # print("the attractor signal value is stable")

                            # add the attractor to the dictionary of output value -> attractors
                            if l_signals_in_attractor[0] == '0':
                                o_output_signal.d_out_value_to_attractor[0].append(o_attractor)
                            elif l_signals_in_attractor[0] == '1':
                                o_output_signal.d_out_value_to_attractor[1].append(o_attractor)
                        # else:
                        #     print("the attractor signal is not stable")
                    if len(set(l_signals_in_local_scene)) == 1:
                        l_signals_for_output.append(l_signals_in_local_scene[0])
                        # print("the scene signal is restricted")
                    else:
                        if len(set(l_signals_in_local_scene)) == 2:
                            l_signals_for_output.extend(l_signals_in_local_scene)
                            # print("the scene signal value is stable")
                        # else:
                        #     print("the scene signal is not stable")
                if len(set(l_signals_for_output)) == 1:
                    o_output_signal.kind_signal = 1
                    # print("the output signal is restricted")
                elif len(set(l_signals_for_output)) == 2:
                    o_output_signal.kind_signal = 3
                    # print("the output signal is stable")
                else:
                    o_output_signal.kind_signal = 4
                    print("THE SCENE SIGNAL IS NOT STABLE. THIS CBN DONT HAVE STABLE ATTRACTOR FIELDS")

            # # print all the kinds of the signals
            # CustomText.print_duplex_line()
            # print("RESUME")
            # print("Network:", l_var_intern.index)
            # for o_directed_edge in self.l_directed_edges:
            #     print(o_directed_edge.index_variable_signal, ":", o_directed_edge.kind_signal)

            # Update the weights of the nodes
            # Add the output network to the list of modified networks
            l_modified_edges = CBN.find_input_edges_by_network_index(o_local_network.index,
                                                                     self.l_directed_edges)
            for o_edge in l_modified_edges:
                modified_network_index = o_edge.output_local_network
                # print("Network", modified_network_index)
                # print("Relation:", o_edge.input_local_network, "->", o_edge.output_local_network)
                weight = 0
                l_edges = CBN.find_input_edges_by_network_index(o_edge.output_local_network,
                                                                self.l_directed_edges)
                for o_updated_edge in l_edges:
                    weight = weight + o_updated_edge.kind_signal
                # print("New weight:", weight)
                o_custom_heap.update_node(o_edge.output_local_network, weight)

            # print("INITIAL HEAP")
            # print(initial_heap)
            # print("UPDATE HEAP")
            # print(o_custom_heap.get_indexes())
            # print("empty heap")
            # print("The Local attractors are computed")
        print("ALL THE ATTRACTORS ARE COMPUTED")

    def find_local_attractors_parallel(self):
        pass

    def find_compatible_pairs(self):
        CustomText.print_duplex_line()
        print("FIND COMPATIBLE ATTRACTOR PAIRS")

        # generate the pairs using the output signal
        l_pairs = []
        # for every local network finds compatible attractor pairs
        for o_local_network in self.l_local_networks:
            # print("----------------------------------------")
            # print("NETWORK -", l_var_intern.index)
            # find the output edges from the local network
            l_output_edges = self.get_output_edges_by_network_index(o_local_network.index)
            # find the pairs for every signal
            for o_output_signal in l_output_edges:
                # print("-------------------------------")
                # print("OUTPUT SIGNAL -", o_output_signal.index_variable)
                # Show the attractors by value of output signal
                # o_output_signal.show_v_output_signal_attractor()
                # coupling the attractors pairs by the output signal
                l_attractors_input_0 = o_output_signal.d_out_value_to_attractor[0]
                l_attractors_input_1 = o_output_signal.d_out_value_to_attractor[1]
                l_pairs_edge_0 = []
                l_pairs_edge_1 = []

                # print("-------------------------------")
                # print("INPUT ATTRACTOR LIST")
                # search the values for every signal
                for signal_value in o_output_signal.d_out_value_to_attractor.keys():
                    # print("-------------------------------")
                    # print("Coupling signal value -", signal_value)
                    # find the attractors that generated by this signal
                    l_attractors_output = []
                    # select the attractor who generated the output value of the signal
                    for o_attractor in self.get_attractors_by_input_signal_value(o_output_signal.index_variable,
                                                                                 signal_value):
                        l_attractors_output.append(o_attractor)
                        # o_attractor.show()
                    if signal_value == 0:
                        l_pairs_edge_0 = list(itertools.product(l_attractors_input_0, l_attractors_output))
                    elif signal_value == 1:
                        l_pairs_edge_1 = list(itertools.product(l_attractors_input_1, l_attractors_output))
                # Join the two list in only one
                o_output_signal.d_comp_pairs_attractors_by_value[0] = l_pairs_edge_0
                o_output_signal.d_comp_pairs_attractors_by_value[1] = l_pairs_edge_1
        print("END FIND ATTRACTOR PAIRS")

    def order_edges_by_compatibility(self):

        def is_compatible(l_group_base, o_group):
            for aux_par in l_group_base:
                if (aux_par.input_local_network == o_group.input_local_network or
                        aux_par.input_local_network == o_group.output_local_network):
                    return True
                elif (aux_par.output_local_network == o_group.output_local_network or
                      aux_par.output_local_network == o_group.input_local_network):
                    return True
            return False

        # Order the groups of compatible pairs
        l_base = [self.l_directed_edges[0]]
        aux_l_rest_groups = self.l_directed_edges[1:]
        for v_group in aux_l_rest_groups:
            if is_compatible(l_base, v_group):
                l_base.append(v_group)
            else:
                aux_l_rest_groups.remove(v_group)
                aux_l_rest_groups.append(v_group)
        self.l_directed_edges = [self.l_directed_edges[0]] + aux_l_rest_groups
        # print("Directed Edges ordered.")

    def find_attractor_fields(self):
        """
        Assembles compatible attractor fields.

        Args:
          List of compatible attractor pairs.

        Returns:
          List of attractor fields.
        """

        def evaluate_pair(base_pairs, candidate_pair):
            """
            Checks if a candidate attractor pair is compatible with a base attractor pair.

            Args:
              base_pairs: Base attractor pairs.
              candidate_pair: Candidate attractor pair.

            Returns:
              Boolean value of True or False.
            """

            # Extract the RDDs from each attractor pair.
            # print("Base pair")
            # print(base_pair)
            base_attractor_pairs = [attractor for pair in base_pairs for attractor in pair]
            # for o_attractor in base_attractor_pairs:
            #     print("Network:", o_attractor.network_index)
            #     print(o_attractor)

            # print("Base List")
            # print(base_attractor_pairs)

            # generate the already networks visited
            l_already_networks = []
            for o_attractor in base_attractor_pairs:
                l_already_networks.append(o_attractor.network_index)
            l_already_networks = set(l_already_networks)

            # Check if any RDD from the candidate attractor pair is present in the RDDs from the base attractor pair.
            double_check = 0
            for candidate_attractor in candidate_pair:
                # print(base_attractor_pairs)
                # print("candidate attractor")
                # print(candidate_attractor)
                if candidate_attractor.network_index in l_already_networks:
                    if candidate_attractor in base_attractor_pairs:
                        double_check = double_check + 1
                else:
                    double_check = double_check + 1
            if double_check == 2:
                return True
            else:
                return False

        def cartesian_product_mod(base_pairs, candidate_pairs):
            """
            Performs the modified Cartesian product the attractor pairs lists.

            Args:
              base_pairs: List of base attractor pairs.
              candidate_pairs: List of candidate attractor pairs.

            Returns:
              List of candidate attractor fields.
            """

            # Initialize the list of candidate attractor fields.
            field_pair_list = []

            # Iterate over the base attractor pairs.
            for base_pair in base_pairs:
                # Iterate over the candidate attractor pairs.
                for candidate_pair in candidate_pairs:
                    # CustomText.print_simple_line()
                    # print("Evaluate Candidate")
                    # show the base
                    # print("Base")
                    # if isinstance(base_pair, list):
                    #     for pair in base_pair:
                    #         pair[0].show()
                    #         pair[1].show()
                    # elif isinstance(base_pair, tuple):
                    #     base_pair[0].show()
                    #     base_pair[1].show()
                    # else:
                    #     raise TypeError("Unsupported base_pair type.")
                    # show the candidate
                    # print("Candidate")
                    # candidate_pair[0].show()
                    # candidate_pair[1].show()

                    # Check if the candidate attractor pair is compatible with the base attractor pair.
                    if isinstance(base_pair, tuple):
                        base_pair = [base_pair]
                    # Evaluate if the pair is compatible with the base
                    if evaluate_pair(base_pair, candidate_pair):
                        # print("compatible pair")
                        new_pair = base_pair + [candidate_pair]
                        # Add the new attractor pair to the list of candidate attractor fields.
                        field_pair_list.append(new_pair)
                    # else:
                    #   print("incompatible pair")
            return field_pair_list

        CustomText.print_duplex_line()
        print("FIND ATTRACTOR FIELDS")

        # Order the edges by compatibility
        self.order_edges_by_compatibility()

        # generate a base list of the pairs
        l_base = self.l_directed_edges[:2]

        # generate the list of pairs made with 0 or 1
        l_base_pairs = l_base[0].d_comp_pairs_attractors_by_value[0] + l_base[0].d_comp_pairs_attractors_by_value[1]

        # for every edge make the union to the base
        for o_directed_edge in self.l_directed_edges[1:]:
            l_candidate_pairs = o_directed_edge.d_comp_pairs_attractors_by_value[0] + \
                                o_directed_edge.d_comp_pairs_attractors_by_value[1]
            # join the base list with the new directed edge
            l_base_pairs = cartesian_product_mod(l_base_pairs, l_candidate_pairs)

        CustomText.print_simple_line()
        print("Number of attractor fields found:", len(l_base_pairs))
        self.l_attractor_fields = l_base_pairs

    # SHOW FUNCTIONS
    @staticmethod
    def show_allowed_topologies():
        # allowed topologies
        allowed_topologies = {
            1: "complete_graph",
            2: "binomial_tree",
            3: "cycle_graph",
            4: "path_graph",
            5: "gn_graph",
            6: "gnc_graph",
            7: "linear_graph"
        }
        for key, value in allowed_topologies.items():
            print(key, "-", value)

    def show_cbn_graph(self):
        G = nx.DiGraph()
        l_networks = []
        for o_edge in self.l_directed_edges:
            l_networks.append((o_edge.input_local_network, o_edge.output_local_network))
        G.add_edges_from(l_networks)
        nx.draw(G)

    def show_attractor_pairs(self):
        CustomText.print_duplex_line()
        print("LIST OF THE COMPATIBLE ATTRACTOR PAIRS")
        for o_directed_edge in self.l_directed_edges:
            CustomText.print_simple_line()
            print("Edge: ", o_directed_edge.output_local_network, "->", o_directed_edge.input_local_network)
            for key in o_directed_edge.d_comp_pairs_attractors_by_value.keys():
                CustomText.print_simple_line()
                print("Coupling Variable:", o_directed_edge.index_variable, "Scene:", key)
                for o_pair in o_directed_edge.d_comp_pairs_attractors_by_value[key]:
                    o_pair[0].show_short()
                    o_pair[1].show_short()

    def show_directed_edges(self):
        CustomText.print_duplex_line()
        print("SHOW THE DIRECTED EDGES OF THE CBN")
        for o_directed_edge in self.l_directed_edges:
            o_directed_edge.show()

    def show_coupled_signals_kind(self):
        CustomText.print_duplex_line()
        print("SHOW THE COUPLED SIGNALS KINDS")
        n_restricted_signals = 0
        for o_directed_edge in self.l_directed_edges:
            print("SIGNAL:", o_directed_edge.index_variable,
                  "RELATION:", o_directed_edge.output_local_network, "->", o_directed_edge.input_local_network,
                  "KIND:", o_directed_edge.kind_signal, "-", o_directed_edge.d_kind_signal[o_directed_edge.kind_signal])
            if o_directed_edge.kind_signal == 1:
                n_restricted_signals = n_restricted_signals + 1
                print("RESTRICTED SIGNAL")
        print("Number of restricted signals :", n_restricted_signals)

    def show_cbn(self):
        CustomText.print_duplex_line()
        print("CBN description")
        l_local_networks_indexes = [o_local_network.index for o_local_network in self.l_local_networks]
        CustomText.print_simple_line()
        print("Local Networks:", l_local_networks_indexes)
        for o_local_network in self.l_local_networks:
            o_local_network.show()
        CustomText.print_simple_line()
        print("Directed edges:")
        for o_directed_edge in self.l_directed_edges:
            o_directed_edge.show()

    def show_attractors(self):
        for o_network in self.l_local_networks:
            CustomText.print_duplex_line()
            print("Network:", o_network.index)
            for o_scene in o_network.l_local_scenes:
                CustomText.print_simple_line()
                print("Network:", o_network.index, "- Scene:", o_scene.l_values)
                print("Attractors number:", len(o_scene.l_attractors))
                for o_attractor in o_scene.l_attractors:
                    CustomText.print_simple_line()
                    for o_state in o_attractor.l_states:
                        print(o_state.l_variable_values)

    def show_global_scenes(self):
        CustomText.print_duplex_line()
        print("LIST OF GLOBAL SCENES")
        for o_global_scene in self.l_global_scenes:
            o_global_scene.show()

    def show_attractors_fields(self):
        CustomText.print_duplex_line()
        print("Show the list of attractor fields")
        print("Number Stable Attractor Fields:", len(self.l_attractor_fields))
        for attractor_field in self.l_attractor_fields:
            CustomText.print_simple_line()
            for pair in attractor_field:
                pair[0].show()
                pair[1].show()

    def show_resume(self):
        CustomText.print_duplex_line()
        print("CBN Resume Indicators")
        print("n_local_attractors", self.get_n_local_attractors())
        print("n_pair_attractors", self.get_n_pair_attractors())
        print("n_attractor_fields", self.get_n_attractor_fields())

    # GET FUNCTIONS
    def get_network_by_index(self, index):
        for o_local_network in self.l_local_networks:
            if o_local_network.index == index:
                return o_local_network

    def get_input_edges_by_network_index(self, index):
        l_input_edges = []
        for o_directed_edge in self.l_directed_edges:
            if o_directed_edge.input_local_network == index:
                l_input_edges.append(o_directed_edge)
        return l_input_edges

    def get_output_edges_by_network_index(self, index):
        l_output_edges = []
        for o_directed_edge in self.l_directed_edges:
            if o_directed_edge.output_local_network == index:
                l_output_edges.append(o_directed_edge)
        return l_output_edges

    def get_index_networks(self):
        indexes_networks = []
        for i_network in self.l_local_networks:
            indexes_networks.append(i_network)
        return indexes_networks

    def get_attractors_by_input_signal_value(self, index_variable_signal, signal_value):
        l_attractors = []
        for o_local_network in self.l_local_networks:
            for scene in o_local_network.l_local_scenes:
                # Validate if the scene have signals or not
                if scene.l_values is not None:
                    if index_variable_signal in scene.l_index_signals:
                        pos = scene.l_index_signals.index(index_variable_signal)
                        if scene.l_values[pos] == str(signal_value):
                            l_attractors = l_attractors + scene.l_attractors
        return l_attractors

    def get_n_local_attractors(self):
        res = 0
        for o_local_network in self.l_local_networks:
            for o_scene in o_local_network.l_local_scenes:
                res = res + len(o_scene.l_attractors)
        return res

    def get_n_pair_attractors(self):
        res = 0
        for o_directed_edge in self.l_directed_edges:
            res += len(o_directed_edge.d_comp_pairs_attractors_by_value[0])
            res += len(o_directed_edge.d_comp_pairs_attractors_by_value[1])
        return res

    def get_n_attractor_fields(self):
        return len(self.l_attractor_fields)


class PathCircleTemplate:
    def __init__(self):
        pass

    @staticmethod
    def generate_aleatory_template(n_var_network):
        """
        Generates aleatory template for a local network
        :param n_var_network:
        :return: Dictionary of cnf function for variable and list of exit variables
        """

        # basic properties
        index = 0
        l_var_intern = list(range(n_var_network + 1, (n_var_network * 2) + 1))
        l_var_exit = random.sample(range(1, n_var_network + 1), 2)
        l_var_external = [n_var_network * 2 + 1]

        # calculate properties
        l_var_total = l_var_intern + l_var_external

        # generate the aleatory dynamic
        d_variable_cnf_function = {}
        b_flag = True
        while b_flag:
            for i_variable in l_var_intern:
                # generate cnf function
                d_variable_cnf_function[i_variable] = random.sample(l_var_total, 3)
                d_variable_cnf_function[i_variable] = [
                    [-element if random.choice([True, False]) else element for element
                     in d_variable_cnf_function[i_variable]]]
            # check if any function has the coupling signal
            for key, value in d_variable_cnf_function.items():
                if l_var_external[0] or -l_var_external[0] in value:
                    b_flag = False

        return d_variable_cnf_function, l_var_exit

    @staticmethod
    def get_output_variables_from_template(i_local_network, l_local_networks, l_var_exit):
        # select the internal variables
        l_variables = []
        for o_local_network in l_local_networks:
            if o_local_network.index == i_local_network:
                # select the specific variables from variable list intern
                for position in l_var_exit:
                    l_variables.append(o_local_network.l_var_intern[position - 1])

        return l_variables

    @staticmethod
    def update_clause_from_template(l_local_networks, o_local_network, i_local_variable, d_variable_cnf_function,
                                    l_directed_edges, v_topology):
        """
        update clause from template
        :param v_topology:
        :param l_local_networks:
        :param o_local_network:
        :param i_local_variable:
        :param d_variable_cnf_function:
        :return: l_clauses_node
        """

        l_indexes_directed_edges = []
        for o_directed_edge in l_directed_edges:
            l_indexes_directed_edges.append(o_directed_edge.index_variable)

        # find the correct cnf function for the variables
        n_local_variables = len(l_local_networks[0].l_var_intern)
        i_template_variable = i_local_variable - ((o_local_network.index - 1) * n_local_variables) + n_local_variables
        pre_l_clauses_node = d_variable_cnf_function[i_template_variable]

        print("Local Variable index:", i_local_variable)
        print("Template Variable index:", i_template_variable)
        print("CNF Function:", pre_l_clauses_node)

        # for every pre-clause update the variables of the cnf function
        l_clauses_node = []
        for pre_clause in pre_l_clauses_node:
            # update the number of the variable
            l_clause = []
            for template_value in pre_clause:
                # evaluate if the topology is 1_linear(4) and is the first local network and not in the list of dictionary
                if (v_topology == 4 and o_local_network.index == 1
                        and abs(template_value) not in list(d_variable_cnf_function.keys())):
                    continue
                else:
                    # save the symbol (+ or -) of the value True for "+" and False for "-"
                    b_symbol = True
                    if template_value < 0:
                        b_symbol = False
                    # replace the value with the variable index
                    local_value = abs(template_value) + (
                            (o_local_network.index - 3) * n_local_variables) + n_local_variables
                    # analyzed if the value is an external value,searching the value in the list of intern variables
                    if local_value not in o_local_network.l_var_intern:
                        # print(o_local_network.l_var_intern)
                        # print(o_local_network.l_var_exterm)
                        # print(local_value)
                        local_value = o_local_network.l_var_exterm[0]
                    # add the symbol to the value
                    if not b_symbol:
                        local_value = -local_value
                    # add the value to the local clause
                    l_clause.append(local_value)

            # add the clause to the list of clauses
            l_clauses_node.append(l_clause)

        print(i_local_variable, ":", l_clauses_node)
        return l_clauses_node

    @staticmethod
    def generate_local_dynamic_with_template(l_local_networks, l_directed_edges, d_variable_cnf_function, v_topology):
        """
        GENERATE THE DYNAMICS OF EACH LOCAL NETWORK
        :param v_topology:
        :param l_local_networks:
        :param l_directed_edges:
        :param d_variable_cnf_function:
        :return: l_local_networks updated
        """
        number_max_of_clauses = 2
        number_max_of_literals = 3

        # generate an auxiliary list to modify the variables
        l_local_networks_updated = []

        # update the dynamic for every local network
        for o_local_network in l_local_networks:
            CustomText.print_simple_line()
            print("Local Network:", o_local_network.index)

            # find the directed edges by network index
            l_input_signals_by_network = CBN.find_input_edges_by_network_index(index=o_local_network.index,
                                                                               l_directed_edges=l_directed_edges)

            # # add the variable index of the directed edges
            # for o_signal in l_input_signals_by_network:
            #     o_local_network.l_var_exterm.append(o_signal.index_variable)
            # o_local_network.l_var_total = o_local_network.l_var_intern + o_local_network.l_var_exterm

            # generate the function description of the variables
            des_funct_variables = []
            # generate clauses for every local network adapting the template
            for i_local_variable in o_local_network.l_var_intern:
                CustomText.print_simple_line()
                # adapting the clause template to the specific variable
                l_clauses_node = PathCircleTemplate.update_clause_from_template(l_local_networks=l_local_networks,
                                                                                o_local_network=o_local_network,
                                                                                i_local_variable=i_local_variable,
                                                                                d_variable_cnf_function=d_variable_cnf_function,
                                                                                l_directed_edges=l_directed_edges,
                                                                                v_topology=v_topology)
                # generate an internal variable from satispy
                o_variable_model = InternalVariable(index=i_local_variable,
                                                    cnf_function=l_clauses_node)
                # adding the description in functions of every variable
                des_funct_variables.append(o_variable_model)

            # adding the local network to a list of local networks
            o_local_network.des_funct_variables = des_funct_variables.copy()
            l_local_networks_updated.append(o_local_network)
            print("Local network created :", o_local_network.index)
            CustomText.print_simple_line()

        # actualized the list of local networks
        return l_local_networks_updated

    @staticmethod
    def get_last_variable(l_local_networks):
        """
        search the last variable from the local network variables
        :param l_local_networks:
        :return:
        """
        last_index_variable = l_local_networks[-1].l_var_intern[-1]
        return last_index_variable

    @staticmethod
    def generate_cbn_from_template(v_topology, d_variable_cnf_function, l_var_exit, n_local_networks, n_var_network):
        """
        Generate a special CBN

        Args:
            v_topology:
            d_variable_cnf_function:
            l_var_exit:
            n_local_networks:
        Returns:
            A CBN generated from a template
        """

        # generate the local networks with the indexes and variables (without relations or dynamics)
        l_local_networks = CBN.generate_local_networks_indexes_variables(n_local_networks=n_local_networks,
                                                                         n_var_network=n_var_network)

        # generate the directed edges between the local networks
        l_directed_edges = []

        # generate the CBN topology
        l_relations = CBN.generate_cbn_topology(n_nodes=n_local_networks,
                                                v_topology=v_topology)

        # Get the last index of the variables for the indexes of the directed edges
        i_last_variable = PathCircleTemplate.get_last_variable(l_local_networks=l_local_networks) + 1

        # generate the directed edges given the last variable generated and the selected output variables
        for relation in l_relations:
            output_local_network = relation[0]
            input_local_network = relation[1]

            # get the output variables from template
            l_output_variables = PathCircleTemplate.get_output_variables_from_template(output_local_network,
                                                                                       l_local_networks, l_var_exit)

            # generate the coupling function
            coupling_function = " " + " ∨ ".join(list(map(str, l_output_variables))) + " "
            # generate the Directed-Edge object
            o_directed_edge = DirectedEdge(index_variable_signal=i_last_variable,
                                           input_local_network=input_local_network,
                                           output_local_network=output_local_network,
                                           l_output_variables=l_output_variables,
                                           coupling_function=coupling_function)
            i_last_variable += 1
            # add the directed-edge object to the list
            l_directed_edges.append(o_directed_edge)

        # Process the coupling signals for every local network
        for o_local_network in l_local_networks:
            # find the signals for every local network
            l_input_signals = CBN.find_input_edges_by_network_index(index=o_local_network.index,
                                                                    l_directed_edges=l_directed_edges)
            # process the input signals of the local network
            o_local_network.process_input_signals(l_input_signals=l_input_signals)

        # generate dynamic of the local networks with template
        l_local_networks = PathCircleTemplate.generate_local_dynamic_with_template(l_local_networks=l_local_networks,
                                                                                   l_directed_edges=l_directed_edges,
                                                                                   d_variable_cnf_function=d_variable_cnf_function,
                                                                                   v_topology=v_topology)

        # generate the special coupled boolean network
        o_special_cbn = CBN(l_local_networks=l_local_networks,
                            l_directed_edges=l_directed_edges)

        return o_special_cbn




"""
Experiment 5 - Test the path and 2_ring structures 
using aleatory generated template for the local network 
"""


@profile
def run_script():
    # experiment parameters
    N_SAMPLES = 20
    N_LOCAL_NETWORKS_MIN = 10
    N_LOCAL_NETWORKS_MAX = 12
    N_VAR_NETWORK = 5
    N_OUTPUT_VARIABLES = 2
    N_INPUT_VARIABLES = 2
    # V_TOPOLOGY = 3  # cycle graph
    N_CLAUSES_FUNCTION = 2
    N_DIRECTED_EDGES = 1

    # verbose parameters
    SHOW_MESSAGES = True

    # Begin the Experiment
    # Capture the time for all the experiment
    v_begin_exp = time.time()

    # generate the experiment path and save the data in csv
    path = ("exp5_aleatory_linear_circle_"
            + str(N_LOCAL_NETWORKS_MIN) + "_"
            + str(N_LOCAL_NETWORKS_MAX)
            + "_" + str(N_SAMPLES) + ".csv")

    # Erase the file if exists
    if os.path.exists(path):
        os.remove(path)
        print("Existing file deleted:", path)

    # Begin the process
    for n_local_networks in range(N_LOCAL_NETWORKS_MIN, N_LOCAL_NETWORKS_MAX + 1):  # 5
        for i_sample in range(1, N_SAMPLES + 1):  # 1 - 1000 , 1, 2
            # generate the aleatory local network template
            d_variable_cnf_function, l_var_exit = PathCircleTemplate.generate_aleatory_template(
                n_var_network=N_VAR_NETWORK)
            for v_topology in [4, 3]:
                l_data_sample = []
                print("Experiment", i_sample, "of", N_SAMPLES, " TOPOLOGY:", v_topology)

                o_cbn = PathCircleTemplate.generate_cbn_from_template(v_topology=v_topology,
                                                                      d_variable_cnf_function=d_variable_cnf_function,
                                                                      l_var_exit=l_var_exit,
                                                                      n_local_networks=n_local_networks,
                                                                      n_var_network=N_VAR_NETWORK)

                # find attractors
                v_begin_find_attractors = time.time()
                o_cbn.find_local_attractors_optimized()
                v_end_find_attractors = time.time()
                n_time_find_attractors = v_end_find_attractors - v_begin_find_attractors

                # find the compatible pairs
                v_begin_find_pairs = time.time()
                o_cbn.find_compatible_pairs()
                v_end_find_pairs = time.time()
                n_time_find_pairs = v_end_find_pairs - v_begin_find_pairs

                # Find attractor fields
                v_begin_find_fields = time.time()
                o_cbn.find_attractor_fields()
                v_end_find_fields = time.time()
                n_time_find_fields = v_end_find_fields - v_begin_find_fields

                # collect indicators
                d_collect_indicators = {
                    # initial parameters
                    "i_sample": i_sample,
                    "n_local_networks": n_local_networks,
                    "n_var_network": N_VAR_NETWORK,
                    "v_topology": v_topology,
                    "n_output_variables": N_OUTPUT_VARIABLES,
                    "n_clauses_function": N_CLAUSES_FUNCTION,
                    # calculate parameters
                    "n_local_attractors": o_cbn.get_n_local_attractors(),
                    "n_pair_attractors": o_cbn.get_n_pair_attractors(),
                    "n_attractor_fields": o_cbn.get_n_attractor_fields(),
                    # time parameters
                    "n_time_find_attractors": n_time_find_attractors,
                    "n_time_find_pairs": n_time_find_pairs,
                    "n_time_find_fields": n_time_find_fields
                }
                l_data_sample.append(d_collect_indicators)

                # save the collected indicator to analysis
                pf_res = pd.DataFrame(l_data_sample)
                pf_res.reset_index(drop=True, inplace=True)

                # if the file exist, open the 'a' mode (append), else create a new file
                mode = 'a' if os.path.exists(path) else 'w'
                # Add the header only if is a new file
                header = not os.path.exists(path)
                #  save the data in csv file
                pf_res.to_csv(path, mode=mode, header=header, index=False)

                print("Experiment saved in:", path)
                CustomText.print_duplex_line()
            CustomText.print_stars()
        CustomText.print_dollars()

    # Take the time of the experiment
    v_end_exp = time.time()
    v_time_exp = v_end_exp - v_begin_exp
    print("Time experiment (in seconds): ", v_time_exp)

    print("End experiment")


run_script()
