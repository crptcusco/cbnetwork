# external imports
import random

# local imports
from classes.cbnetwork import CBN
from classes.directededge import DirectedEdge
from classes.internalvariable import InternalVariable
from classes.utils.customtext import CustomText


class PathCircleTemplate:
    def __init__(self, n_var_network, d_variable_cnf_function, l_output_var_indexes):
        self.n_var_network = n_var_network
        self.d_variable_cnf_function = d_variable_cnf_function
        self.l_output_var_indexes = l_output_var_indexes

    def show(self):
        print("Template for Path and Circle CBNs")
        print("-" * 80)
        print("Local dynamic:")
        for key, value in self.d_variable_cnf_function.items():
            print(key, ":", value)
        print("Output variables for the coupling signal:")
        print(self.l_output_var_indexes)

    @staticmethod
    def generate_path_circle_template(n_var_network, n_input_variables=2, n_output_variables=2):
        """
        Generates aleatory template for a local network
        :param n_output_variables:
        :param n_input_variables:
        :param n_var_network:
        :return: Dictionary of cnf function for variable and list of exit variables
        """

        # basic properties
        l_internal_var_indexes = list(range(n_var_network + 1, (n_var_network * 2) + 1))
        l_output_var_indexes = random.sample(range(1, n_var_network + 1), n_output_variables)
        l_input_coupling_signal_indexes = [n_var_network * 2 + 1]

        # calculate properties
        l_var_total_indexes = l_internal_var_indexes + l_input_coupling_signal_indexes

        # generate the aleatory dynamic
        d_variable_cnf_function = {}

        # select the internal variables that are going to have external variables
        internal_vars_for_external = random.sample(l_internal_var_indexes, n_input_variables)

        # generate cnf function for every internal variable
        for i_variable in l_internal_var_indexes:
            # evaluate if the variable is in internal_vars_for_external
            if i_variable in internal_vars_for_external:
                external_flag = False
                while not external_flag:
                    d_variable_cnf_function[i_variable] = [random.sample(l_var_total_indexes, 3)]
                    if any(element in d_variable_cnf_function[i_variable][0] for element in
                           l_input_coupling_signal_indexes):
                        external_flag = True
            else:
                # generate cnf function without external variables
                d_variable_cnf_function[i_variable] = [random.sample(l_internal_var_indexes, 3)]

            # apply negation randomly
            d_variable_cnf_function[i_variable][0] = [
                -element if random.choice([True, False]) else element for element
                in d_variable_cnf_function[i_variable][0]]

        # Generate the object of PathCircleTemplate
        o_path_circle_template = PathCircleTemplate(n_var_network, d_variable_cnf_function, l_output_var_indexes)
        return o_path_circle_template

    def get_output_variables_from_template(self, i_local_network, l_local_networks):
        # select the internal variables
        l_variables = []
        for o_local_network in l_local_networks:
            if o_local_network.index == i_local_network:
                # select the specific variables from variable list intern
                for position in self.l_output_var_indexes:
                    l_variables.append(o_local_network.l_var_intern[position - 1])

        return l_variables

    def update_clause_from_template(self, l_local_networks, o_local_network, i_local_variable, l_directed_edges,
                                    v_topology):
        """
        update clause from template
        :param l_directed_edges:
        :param v_topology:
        :param l_local_networks:
        :param o_local_network:
        :param i_local_variable:
        :return: l_clauses_node
        """

        l_indexes_directed_edges = []
        for o_directed_edge in l_directed_edges:
            l_indexes_directed_edges.append(o_directed_edge.index_variable)

        # find the correct cnf function for the variables
        n_local_variables = len(l_local_networks[0].l_var_intern)
        i_template_variable = i_local_variable - ((o_local_network.index - 1) * n_local_variables) + n_local_variables
        pre_l_clauses_node = self.d_variable_cnf_function[i_template_variable]

        print("Local Variable index:", i_local_variable)
        print("Template Variable index:", i_template_variable)
        print("CNF Function:", pre_l_clauses_node)

        # for every pre-clause update the variables of the cnf function
        l_clauses_node = []
        for pre_clause in pre_l_clauses_node:
            # update the number of the variable
            l_clause = []
            for template_value in pre_clause:
                # evaluate if the topology is linear(4) and is the first local network and not in the list of dictionary
                if (v_topology == 4 and o_local_network.index == 1
                        and abs(template_value) not in list(self.d_variable_cnf_function.keys())):
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

    def generate_local_dynamic_with_template(self, l_local_networks, l_directed_edges, v_topology):
        """
        GENERATE THE DYNAMICS OF EACH LOCAL NETWORK
        :param v_topology:
        :param l_local_networks:
        :param l_directed_edges:
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
                l_clauses_node = self.update_clause_from_template(l_local_networks=l_local_networks,
                                                                  o_local_network=o_local_network,
                                                                  i_local_variable=i_local_variable,
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

    def generate_cbn_from_template(self, v_topology, n_local_networks):
        """
        Generate a special CBN

        Args:
            v_topology: The topology of the CBN cam be 'linear' or 'ring'
            n_local_networks: The number of local networks
        Returns:
            A CBN generated from a template
        """

        # generate the local networks with the indexes and variables (without relations or dynamics)
        l_local_networks = CBN.generate_local_networks_indexes_variables(n_local_networks=n_local_networks,
                                                                         n_var_network=self.n_var_network)

        # generate the directed edges between the local networks
        l_directed_edges = []

        # generate the CBN topology
        l_relations = CBN.generate_global_topology(n_nodes=n_local_networks,
                                                   v_topology=v_topology)

        # Get the last index of the variables for the indexes of the directed edges
        i_last_variable = l_local_networks[-1].l_var_intern[-1] + 1

        # generate the directed edges given the last variable generated and the selected output variables
        i_directed_edge = 1
        for relation in l_relations:
            output_local_network = relation[0]
            input_local_network = relation[1]

            # get the output variables from template
            l_output_variables = self.get_output_variables_from_template(output_local_network,
                                                                         l_local_networks)

            # generate the coupling function
            coupling_function = " " + " ∨ ".join(list(map(str, l_output_variables))) + " "
            # generate the Directed-Edge object
            o_directed_edge = DirectedEdge(index=i_directed_edge,
                                           index_variable_signal=i_last_variable,
                                           input_local_network=input_local_network,
                                           output_local_network=output_local_network,
                                           l_output_variables=l_output_variables,
                                           coupling_function=coupling_function)
            i_last_variable += 1
            i_directed_edge += 1
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
        l_local_networks = self.generate_local_dynamic_with_template(l_local_networks=l_local_networks,
                                                                     l_directed_edges=l_directed_edges,
                                                                     v_topology=v_topology)

        # generate the special coupled boolean network
        o_special_cbn = CBN(l_local_networks=l_local_networks,
                            l_directed_edges=l_directed_edges)

        return o_special_cbn


import random


def generate_random_cnf(variables):
    # Generate a random number of clauses based on the number of variables
    num_clauses = random.randint(1, 2)

    cnf = []

    for _ in range(num_clauses):
        # Generate a random number of literals in each clause, with a maximum of 3
        num_literals = min(random.randint(2, 3), len(variables))

        clause = []
        for _ in range(num_literals):
            # Randomly select a variable from the list
            var = random.choice(variables)
            # Randomly decide whether the literal is negated or not
            if random.choice([True, False]):
                var = -var
            clause.append(var)

        # Remove redundant literals within the clause
        clause = simplify_clause(clause)

        # Make sure the clause is not empty
        if clause:
            cnf.append(clause)

    # Ensure at least one non-empty clause
    if not cnf:
        # If no non-empty clauses, add a random non-empty clause
        var = random.choice(variables)
        if random.choice([True, False]):
            var = -var
        cnf.append([var])

    return cnf


def simplify_clause(clause):
    # Remove duplicate literals
    clause = list(set(clause))

    # Check for complementary literals (e.g., x and -x) and remove both
    simplified_clause = []
    for literal in clause:
        if -literal not in clause:
            simplified_clause.append(literal)

    return simplified_clause


class TopologyTemplate:
    def __init__(self, n_var_network, d_variable_cnf_function, l_output_var_indexes, v_topology):
        self.n_var_network = n_var_network
        self.d_variable_cnf_function = d_variable_cnf_function
        self.l_output_var_indexes = l_output_var_indexes
        self.v_topology = v_topology

    def show(self):
        print("Template for Aleatory CBNs")
        print("-" * 80)
        print("Local dynamic:")
        for key, value in self.d_variable_cnf_function.items():
            print(key, ":", value)
        print("Output variables for coupling signal:", self.l_output_var_indexes)

    @staticmethod
    def generate_aleatory_template(n_var_network=5, n_input_variables=2, n_output_variables=2, v_topology=1):
        """
        Generates aleatory template for a local network
        :param v_topology:
        :param n_output_variables:
        :param n_input_variables:
        :param n_var_network:
        :return: Dictionary of cnf function for variable and list of exit variables
        """

        # basic properties
        l_internal_var_indexes = list(range(n_var_network + 1, (n_var_network * 2) + 1))
        l_output_var_indexes = random.sample(range(1, n_var_network + 1), n_output_variables)
        l_input_coupling_signal_indexes = [n_var_network * 2 + 1]

        # calculate properties
        l_var_total_indexes = l_internal_var_indexes + l_input_coupling_signal_indexes

        # generate the aleatory dynamic
        d_variable_cnf_function = {}

        # generate cnf function for every internal variable using generate_random_cnf
        for i_variable in l_internal_var_indexes:
            d_variable_cnf_function[i_variable] = generate_random_cnf(l_var_total_indexes)

        # Generate the object of AleatoryTemplate
        o_aleatory_template = TopologyTemplate(n_var_network=n_var_network,
                                               d_variable_cnf_function=d_variable_cnf_function,
                                               l_output_var_indexes=l_output_var_indexes,
                                               v_topology=v_topology)
        return o_aleatory_template

    def get_output_variables_from_template(self, i_local_network, l_local_networks):
        # select the internal variables
        l_variables = []
        for o_local_network in l_local_networks:
            if o_local_network.index == i_local_network:
                # select the specific variables from variable list intern
                for position in self.l_output_var_indexes:
                    l_variables.append(o_local_network.l_var_intern[position - 1])

        return l_variables

    def update_clause_from_template(self, l_local_networks, o_local_network, i_local_variable, l_directed_edges,
                                    v_topology):
        """
        update clause from template
        :param l_directed_edges:
        :param v_topology:
        :param l_local_networks:
        :param o_local_network:
        :param i_local_variable:
        :return: l_clauses_node
        """

        l_indexes_directed_edges = []
        for o_directed_edge in l_directed_edges:
            l_indexes_directed_edges.append(o_directed_edge.index_variable)

        # find the correct cnf function for the variables
        n_local_variables = len(l_local_networks[0].l_var_intern)
        i_template_variable = i_local_variable - ((o_local_network.index - 1) * n_local_variables) + n_local_variables
        pre_l_clauses_node = self.d_variable_cnf_function[i_template_variable]

        print("Local Variable index:", i_local_variable)
        print("Template Variable index:", i_template_variable)
        print("CNF Function:", pre_l_clauses_node)

        # for every pre-clause update the variables of the cnf function
        l_clauses_node = []
        for pre_clause in pre_l_clauses_node:
            # update the number of the variable
            l_clause = []
            for template_value in pre_clause:
                # evaluate if the topology is aleatory(6) and not in the list of dictionary
                if (v_topology == 6 and abs(template_value) not in list(self.d_variable_cnf_function.keys())):
                    continue
                else:
                    # save the symbol (+ or -) of the value True for "+" and False for "-"
                    b_symbol = True
                    if template_value < 0:
                        b_symbol = False
                    # replace the value with the variable index
                    local_value = abs(template_value) + (
                            (o_local_network.index - 3) * n_local_variables) + n_local_variables
                    # analyzed if the value is an external value, searching the value in the list of intern variables
                    if local_value not in o_local_network.l_var_intern:
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

    def generate_local_dynamic_with_template(self, l_local_networks, l_directed_edges, v_topology):
        """
        GENERATE THE DYNAMICS OF EACH LOCAL NETWORK
        :param v_topology:
        :param l_local_networks:
        :param l_directed_edges:
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

            # generate the function description of the variables
            des_funct_variables = []
            # generate clauses for every local network adapting the template
            for i_local_variable in o_local_network.l_var_intern:
                CustomText.print_simple_line()
                # adapting the clause template to the specific variable
                l_clauses_node = self.update_clause_from_template(l_local_networks=l_local_networks,
                                                                  o_local_network=o_local_network,
                                                                  i_local_variable=i_local_variable,
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

    def generate_cbn_from_template(self, v_topology, n_local_networks):
        """
        Generate a special CBN

        Args:
            v_topology: The topology of the CBN cam be 'aleatory'
            n_local_networks: The number of local networks
        Returns:
            A CBN generated from a template
        """

        # generate the local networks with the indexes and variables (without relations or dynamics)
        l_local_networks = CBN.generate_local_networks_indexes_variables(n_local_networks=n_local_networks,
                                                                         n_var_network=self.n_var_network)

        # generate the directed edges between the local networks
        l_directed_edges = []

        # generate the CBN topology
        l_relations = CBN.generate_global_topology(n_nodes=n_local_networks,
                                                   v_topology=v_topology)

        # Get the last index of the variables for the indexes of the directed edges
        i_last_variable = l_local_networks[-1].l_var_intern[-1] + 1

        # generate the directed edges given the last variable generated and the selected output variables
        i_directed_edge = 1
        for relation in l_relations:
            output_local_network = relation[0]
            input_local_network = relation[1]

            # get the output variables from template
            l_output_variables = self.get_output_variables_from_template(output_local_network,
                                                                         l_local_networks)

            # generate the coupling function
            coupling_function = " " + " ∨ ".join(list(map(str, l_output_variables))) + " "
            # generate the Directed-Edge object
            o_directed_edge = DirectedEdge(index=i_directed_edge,
                                           index_variable_signal=i_last_variable,
                                           input_local_network=input_local_network,
                                           output_local_network=output_local_network,
                                           l_output_variables=l_output_variables,
                                           coupling_function=coupling_function)
            i_last_variable += 1
            i_directed_edge += 1
            # add the directed-edge object to the list
            l_directed_edges.append(o_directed_edge)
            # o_directed_edge.show()

        # for o_local_network in l_local_networks:
        #     o_local_network.show()
        # print(l_relations)

        # Process the coupling signals for every local network
        for o_local_network in l_local_networks:
            # find the signals for every local network
            l_input_signals = CBN.find_input_edges_by_network_index(index=o_local_network.index,
                                                                    l_directed_edges=l_directed_edges)
            # process the input signals of the local network
            o_local_network.process_input_signals(l_input_signals=l_input_signals)
            # o_local_network.show()
            # print([element.index_variable for element in l_input_signals])

        # generate dynamic of the local networks with template
        l_local_networks = self.generate_local_dynamic_with_template(l_local_networks=l_local_networks,
                                                                     l_directed_edges=l_directed_edges,
                                                                     v_topology=v_topology)

        for o_local_network in l_local_networks:
            o_local_network.show()

        # # generate the special coupled boolean network
        # o_special_cbn = CBN(l_local_networks=l_local_networks,
        #                     l_directed_edges=l_directed_edges)
        #
        # return o_special_cbn
