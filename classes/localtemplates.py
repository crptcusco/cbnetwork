# external imports
import random

# local imports
from classes.cbnetwork import CBN
from classes.directededge import DirectedEdge
from classes.globaltopology import GlobalTopology
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
        print("Template Function:", pre_l_clauses_node)

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

        print("Local Variable Index:", i_local_variable)
        print("CNF Function:", l_clauses_node)

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

        # generate the Global Topology Object
        o_global_topology = GlobalTopology(v_topology=v_topology, n_nodes=n_local_networks)
        o_global_topology.generate_networkx_graph()

        l_relations = o_global_topology.get_edges()

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


class CNFList:
    @staticmethod
    def generate_cnf(l_inter_vars, input_coup_sig_index, max_clauses=2, max_literals=3):
        num_clauses = random.randint(1, max_clauses)  # Ensure at least one clause is generated
        l_cnf = []

        # Generate the clause for external signals
        if input_coup_sig_index is not None:
            var = input_coup_sig_index
            if random.choice([True, False]):
                var = -var
            l_cnf.append([var])

        for _ in range(num_clauses):
            clause = []
            while len(clause) < max_literals:
                var = random.choice(l_inter_vars)
                if var != input_coup_sig_index and -var != input_coup_sig_index:
                    if random.choice([True, False]):
                        var = -var
                    clause.append(var)

            # Remove redundant literals within the clause
            clause = CNFList.simplify_clause(clause)

            # Ensure the clause is not empty and has at least one literal
            if clause:
                l_cnf.append(clause)

        # Remove empty clauses
        l_cnf = [clause for clause in l_cnf if clause]

        # Remove duplicate clauses
        l_cnf = CNFList.remove_duplicates(l_cnf)

        return l_cnf

    @staticmethod
    def simplify_clause(clause):
        # Remove duplicate literals
        clause = list(set(clause))

        # Check for complementary literals (e.g., x and -x) and remove both
        simplified_clause = []
        for literal in clause:
            if -literal not in clause:
                simplified_clause.append(literal)

        return simplified_clause

    @staticmethod
    def remove_duplicates(l_cnf):
        # Convert each clause to a tuple and create a set to remove duplicates
        unique_clauses = set(tuple(sorted(clause)) for clause in l_cnf)
        # Convert the unique tuples back to lists
        return [list(clause) for clause in unique_clauses]


class LocalNetworkTemplate:
    def __init__(self, v_topology, n_var_network, l_edges):
        self.n_var_network = n_var_network
        self.v_topology = v_topology
        self.n_edges = len(l_edges)

        self.l_output_var_indexes = []
        self.d_variable_cnf_function = []

        if v_topology == 2:
            # generate Aleatory template
            self.generate_aleatory_template()

        if v_topology == 3:
            # generate circle template
            self.generate_circle_template()

        if v_topology == 4:
            # generate linear template
            self.generate_linear_template()



    def generate_aleatory_template(self, n_var_network=5, n_input_variables=2, n_output_variables=2, v_topology=6,
                                   n_edges=None, n_max_of_clauses=2, n_max_of_literals=3):

        # basic properties
        l_internal_var_indexes = list(range(n_var_network + 1, (n_var_network * 2) + 1))

        l_input_coupling_signal_indexes = [n_var_network * 2 + 1]

        # generate cnf function for every internal variable using generate_random_cnf
        # Track the variables that will include the input_coupling_signal_index
        l_input_variables = random.sample(l_internal_var_indexes, n_input_variables)

        # Generate CNF function for every internal variable using generate_random_cnf
        for i_variable in l_internal_var_indexes:
            input_coup_sig_index = None
            if i_variable in l_input_variables:
                input_coup_sig_index = random.choice(l_input_coupling_signal_indexes)

            self.d_variable_cnf_function[i_variable] = CNFList.generate_cnf(l_inter_vars=l_internal_var_indexes,
                                                                       input_coup_sig_index=input_coup_sig_index,
                                                                       max_clauses=n_max_of_clauses,
                                                                       max_literals=n_max_of_literals)

        # # Generate the object of AleatoryTemplate
        #l_output_var_indexes = random.sample(range(1, n_var_network + 1), n_output_variables)
        # o_aleatory_template = LocalNetworkTemplate(n_var_network=n_var_network,
        #                                            d_variable_cnf_function=d_variable_cnf_function,
        #                                            l_output_var_indexes=l_output_var_indexes,
        #                                            v_topology=v_topology,
        #                                            n_edges=n_edges)
        # return o_aleatory_template

    def show(self):
        print("Local Network Template CBNs")
        print("-" * 80)
        print("Local dynamic:")
        for key, value in self.d_variable_cnf_function.items():
            print(key, ":", value)
        print("Output variables for coupling signal:", self.l_output_var_indexes)

    @staticmethod
    def generate_template(n_variables, n_input_variables, n_output_variables):
        # generate the list of variables
        l_variables = list(range(1, n_variables + 1))
        # generate the list of input variables
        l_input_variables = random.sample(l_variables, n_input_variables)
        # generate the list of output variables
        l_output_variables = random.sample(l_variables, n_output_variables)

        o_local_network_template = LocalNetworkTemplate(l_variables, l_input_variables, l_output_variables)
        return o_local_network_template



    def get_output_variables_from_template(self, i_local_network, l_local_networks):
        # select the internal variables
        l_variables = []
        for o_local_network in l_local_networks:
            if o_local_network.index == i_local_network:
                # select the specific variables from variable list intern
                for position in self.l_output_var_indexes:
                    l_variables.append(o_local_network.l_var_intern[position - 1])

        return l_variables

    def update_clause_from_template(self, l_local_networks, o_local_network, i_local_variable, l_directed_edges):
        """
        update clause from template
        :param l_directed_edges:
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
        print("Template Function:", pre_l_clauses_node)

        # for every pre-clause update the variables of the cnf function
        l_clauses_node = []
        for pre_clause in pre_l_clauses_node:
            # update the number of the variable
            l_clause = []
            for template_value in pre_clause:
                # save the symbol (+ or -) of the value True for "+" and False for "-"
                b_symbol = True
                if template_value < 0:
                    b_symbol = False
                # replace the value with the variable index
                local_value = abs(template_value) + (
                        (o_local_network.index - 3) * n_local_variables) + n_local_variables
                # analyzed if the value is an external value, searching the value in the list of intern variables
                if local_value not in o_local_network.l_var_intern:
                    # put all the external variables in a clause
                    l_clause = o_local_network.l_var_exterm
                    continue
                # add the symbol to the value
                if not b_symbol:
                    local_value = -local_value
                # add the value to the local clause
                l_clause.append(local_value)

            # add the clause to the list of clauses
            l_clauses_node.append(l_clause)

        # Drop empty clauses
        l_clauses_node = [clause for clause in l_clauses_node if clause]

        print("Local Variable Index:", i_local_variable)
        print("CNF Function:", l_clauses_node)

        return l_clauses_node

    def generate_local_dynamic_with_template(self, l_local_networks, l_directed_edges, v_topology):
        """
        GENERATE THE DYNAMICS OF EACH LOCAL NETWORK WITH LOCAL NETWORK TEMPLATE
        :param v_topology:
        :param l_local_networks:
        :param l_directed_edges:
        :return: l_local_networks updated
        """
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
                                                                  l_directed_edges=l_input_signals_by_network)
                # generate an internal variable from satispy
                o_variable_model = InternalVariable(index=i_local_variable, cnf_function=l_clauses_node)
                # adding the description in functions of every variable
                des_funct_variables.append(o_variable_model)

            # adding the local network to a list of local networks
            o_local_network.des_funct_variables = des_funct_variables.copy()
            l_local_networks_updated.append(o_local_network)
            print("Local network created :", o_local_network.index)
            CustomText.print_simple_line()

        # actualized the list of local networks
        return l_local_networks_updated

    def generate_cbn_from_template(self, v_topology, n_local_networks, n_edges, o_base_graph=None):
        """
        Generate a special CBN

        Args:
            v_topology: The topology of the CBN cam be 'aleatory'
            n_local_networks: The number of local networks
            n_edges: the number of edges between local networks
            o_base_graph: a graph who is base for the new graph
        Returns:
            A CBN generated from a template

        """

        # generate the local networks with the indexes and variables (without relations or dynamics)
        l_local_networks = CBN.generate_local_networks_indexes_variables(n_local_networks=n_local_networks,
                                                                         n_var_network=self.n_var_network)

        # generate the directed edges between the local networks
        l_directed_edges = []

        # generate the Global Topology Object
        o_global_topology = GlobalTopology(v_topology=v_topology, n_nodes=n_local_networks, n_edges=n_edges,
                                           o_graph=o_base_graph)
        o_global_topology.generate_networkx_graph()

        l_relations = o_global_topology.get_edges()

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

        # add the Global Topology Object
        o_special_cbn.o_global_topology = o_global_topology

        return o_special_cbn


# parameters
N_VARIABLES = 5
N_INPUT_VARIABLES = 2
N_OUTPUT_VARIABLES = 2
V_TOPOLOGY = 2

# Example usage:
l_internal_variables = [1, 2, 3, 4, 5]
input_coupling_signal_index = 6
cnf = CNFList.generate_cnf(l_internal_variables, input_coupling_signal_index)
print(cnf)
