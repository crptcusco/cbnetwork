import re  # analysis of regular expressions
import operator  # unary operator management

from string import ascii_lowercase, ascii_uppercase     # import the list of uppercase and lowercase letters
from itertools import product                           # generate combinations of numbers
from collections import namedtuple                      # structures like trees
from classes.utils.customtext import CustomText         # utils for texts


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
