# internal imports
from classes.globalscene import GlobalScene
from classes.globaltopology import GlobalTopology
from classes.internalvariable import InternalVariable
from classes.localnetwork import LocalNetwork
from classes.directededge import DirectedEdge
from classes.localtemplates import LocalNetworkTemplate
from classes.utils.customtext import CustomText

# external imports
import itertools  # libraries to iterate
import random  # generate random numbers
from random import randint  # generate random numbers integers
from itertools import product  # generate combinations of numbers


class CBN:
    """
    Class representing a Complex Boolean Network (CBN).

    Attributes:
        l_local_networks (list): List of local networks in the CBN.
        l_directed_edges (list): List of directed edges in the CBN.
        d_local_attractors (dict): Dictionary of local attractors.
        d_attractor_pair (dict): Dictionary of attractor pairs.
        d_attractor_fields (dict): Dictionary of attractor fields.
        l_global_scenes (list): List of global scenes.
        o_global_topology (GlobalTopology): Global topology object.
    """

    def __init__(self, l_local_networks, l_directed_edges):
        # basic attributes
        self.l_local_networks = l_local_networks
        self.l_directed_edges = l_directed_edges

        # calculated attributes
        self.d_local_attractors = None
        self.d_attractor_pair = None
        self.d_attractor_fields = None
        self.l_global_scenes = None

        # graphs
        self.o_global_topology = None

    @staticmethod
    def generate_local_networks_indexes_variables(n_local_networks, n_vars_network):
        """
        Generates local networks and their variable indexes.

        Args:
            n_local_networks (int): Number of local networks to generate.
            n_vars_network (int): Number of variables per network.

        Returns:
            list: List of LocalNetwork objects.
        """
        l_local_networks = []
        v_cont_var = 1
        for v_num_network in range(1, n_local_networks + 1):
            # generate the variables of the networks
            l_var_intern = list(range(v_cont_var, v_cont_var + n_vars_network))
            # create the Local Network object
            o_local_network = LocalNetwork(v_num_network, l_var_intern)
            # add the local network object to the list
            l_local_networks.append(o_local_network)
            # update the index of the variables
            v_cont_var += n_vars_network
        return l_local_networks

    @staticmethod
    def generate_directed_edges(i_last_variable, l_local_networks, l_relations, n_output_variables=2):
        """
        Generates directed edges based on local networks and their relations.

        Args:
            i_last_variable (int): Index of the last variable.
            l_local_networks (list): List of LocalNetwork objects.
            l_relations (list): List of tuples representing relations between local networks.
            n_output_variables (int, optional): Number of output variables. Defaults to 2.

        Returns:
            list: List of DirectedEdge objects.
        """
        l_directed_edges = []
        i_directed_edge = i_last_variable + 1
        index_edge = 1

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

                o_directed_edge = DirectedEdge(index_edge, i_directed_edge, o_local_network.index,
                                               o_local_network_co.index, l_output_variables, coupling_function)
                l_directed_edges.append(o_directed_edge)
                i_directed_edge += 1
                index_edge += 1

        return l_directed_edges

    @staticmethod
    def find_input_edges_by_network_index(index, l_directed_edges):
        """
        Finds all input edges for a given network index.

        Args:
            index (int): The index of the local network.
            l_directed_edges (list): List of DirectedEdge objects.

        Returns:
            list: List of DirectedEdge objects that are input edges for the specified network index.
        """
        return [edge for edge in l_directed_edges if edge.input_local_network == index]

    @staticmethod
    def find_output_edges_by_network_index(index, l_directed_edges):
        """
        Finds all output edges for a given network index.

        Args:
            index (int): The index of the local network.
            l_directed_edges (list): List of DirectedEdge objects.

        Returns:
            list: List of DirectedEdge objects that are output edges for the specified network index.
        """
        return [edge for edge in l_directed_edges if edge.output_local_network == index]

    @staticmethod
    def generate_local_networks_variables_dynamic(l_local_networks, l_directed_edges):
        """
        Generates the dynamics for each local network by creating internal variables and their clauses.

        Args:
            l_local_networks (list): List of LocalNetwork objects.
            l_directed_edges (list): List of DirectedEdge objects.

        Returns:
            list: List of updated LocalNetwork objects with dynamic variables.
        """

        # the max number of clauses and literals is fixed in 2 and 3 respectively
        NUMBER_MAX_OF_CLAUSES = 2
        NUMBER_MAX_OF_LITERALS = 3

        l_local_networks_updated = []
        for o_local_network in l_local_networks:
            l_aux_variables = []
            l_input_signals = CBN.find_input_edges_by_network_index(o_local_network.index, l_directed_edges)
            for o_signal in l_input_signals:
                l_aux_variables.append(o_signal.index_variable)
            l_aux_variables.extend(o_local_network.l_var_intern)

            des_funct_variables = []
            for i_local_variable in o_local_network.l_var_intern:
                l_clauses_node = []
                for _ in range(randint(1, NUMBER_MAX_OF_CLAUSES)):
                    v_num_variable = randint(1, NUMBER_MAX_OF_LITERALS)
                    l_literals_variables = random.sample(l_aux_variables, v_num_variable)
                    l_clauses_node.append(l_literals_variables)

                o_internal_variable = InternalVariable(i_local_variable, l_clauses_node)
                des_funct_variables.append(o_internal_variable)

            o_local_network.des_funct_variables = des_funct_variables.copy()
            l_local_networks_updated.append(o_local_network)
            CustomText.make_sub_title(f"Local network created : {o_local_network.index}")

        return l_local_networks_updated

    @staticmethod
    def generate_aleatory_cbn_by_topology(n_local_networks, n_var_network, v_topology, l_edges, n_output_variables=2,
                                          n_input_variables=2, n_edges=None, local_template=None):
        """
        Generates an instance of a CBN.

        Args:
            n_local_networks (int): Number of local networks to generate.
            n_var_network (int): Number of variables per network.
            v_topology (int): Type of topology for the CBN.
            l_edges (list): List of global edges.
            n_output_variables (int): Number of output variables.
            n_input_variables (int): Number of input variables.
            n_edges (int): Number of edges in the CBN.
            local_template (AleatoryTemplate): Template for local networks.

        Returns:
            CBN: The generated CBN object.
        """

        CustomText.make_title('CBN GENERATION')

        # generate the local network dynamic
        # if the template is None generated an aleatory dynamic for the variables
        if local_template is None:
            # generate the local networks with the indexes and variables (without relations or dynamics)
            l_local_networks = CBN.generate_local_networks_indexes_variables(n_local_networks, n_var_network)

            # generate the CBN topology
            o_global_topology = GlobalTopology(v_topology=v_topology, l_edges=l_edges)
            o_global_topology.generate_networkx_graph()
            l_relations = o_global_topology.get_edges()

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

            l_local_networks = CBN.generate_local_networks_variables_dynamic(l_local_networks=l_local_networks,
                                                                             l_directed_edges=l_directed_edges)

            # create the cbn object
            o_cbn = CBN(l_local_networks, l_directed_edges)
            # add the Global Topology Object
            o_cbn.o_global_topology = o_global_topology
            return o_cbn

        # if you have a template we use the template for the variables dynamics
        else:
            o_cbn = local_template.generate_cbn_from_template(v_topology=v_topology,
                                                              n_local_networks=n_local_networks,
                                                              n_edges=n_edges)
            return o_cbn

    def process_output_signals(self):
        # update output signals for every local network
        for o_local_network in self.l_local_networks:
            for t_relation in self.l_directed_edges:
                if o_local_network.l_index == t_relation[1]:
                    o_local_network.l_output_signals.append(t_relation)
                    print(t_relation)

    def update_network_by_index(self, o_local_network_update):
        for i, o_local_network in enumerate(self.l_local_networks):
            if o_local_network.l_index == o_local_network_update.l_index:
                self.l_local_networks[i] = o_local_network_update
                print("Local Network updated")
                return True
        print("ERROR:", "Local Network not found")
        return False

    def find_local_attractors_sequential(self):
        """
        Finds local attractors sequentially.

        Returns:
            str: Update the list of local attractor in the object.
        """
        CustomText.make_title('FIND LOCAL ATTRACTORS')

        for o_local_network in self.l_local_networks:
            l_local_scenes = None
            if len(o_local_network.l_var_exterm) != 0:
                l_local_scenes = list(product(list('01'), repeat=len(o_local_network.l_var_exterm)))
                # calculate the attractors for the node in the top of the  heap
                o_local_network = LocalNetwork.find_local_attractors(o_local_network=o_local_network,
                                                                     l_local_scenes=l_local_scenes)
            else:
                # calculate the attractors for local network without coupling signals
                o_local_network = LocalNetwork.find_local_attractors(o_local_network=o_local_network,
                                                                     l_local_scenes=l_local_scenes)

            # Update the coupling signals to analyzed
            self.process_kind_signal(o_local_network)

        # Put the global index for every local attractor
        i_attractor = 1
        for o_local_network in self.l_local_networks:
            for o_local_scene in o_local_network.l_local_scenes:
                for o_attractor in o_local_scene.l_attractors:
                    o_attractor.g_index = i_attractor
                    i_attractor += 1

        # Generate the attractor dictionary
        self.generate_attractor_dictionary()

        print('Number of local attractors:', i_attractor)
        CustomText.make_sub_sub_title('END FIND LOCAL ATTRACTORS')

    def process_kind_signal(self, o_local_network):
        """
        validate if the output variables by attractor send a fixed value and update kind signals
        and update the kind of signal
        :param o_local_network:
        :return: update kind of signals
        """

        # Find the edges who interact with the local network
        l_directed_edges = CBN.find_output_edges_by_network_index(o_local_network.index, self.l_directed_edges)
        # For every edge who interacts, compute the kind
        for o_output_signal in l_directed_edges:
            l_signals_for_output = []
            for o_local_scene in o_local_network.l_local_scenes:
                l_signals_in_local_scene = []
                for o_attractor in o_local_scene.l_attractors:
                    l_signals_in_attractor = []
                    for o_state in o_attractor.l_states:
                        true_table_index = ""
                        for v_output_variable in o_output_signal.l_output_variables:
                            pos = o_local_network.l_var_total.index(v_output_variable)
                            value = o_state.l_variable_values[pos]
                            true_table_index = true_table_index + str(value)
                        output_value_state = o_output_signal.true_table[true_table_index]
                        l_signals_in_attractor.append(output_value_state)
                    if len(set(l_signals_in_attractor)) == 1:
                        l_signals_in_local_scene.append(l_signals_in_attractor[0])
                        # add the attractor to the dictionary of output value -> attractors
                        if l_signals_in_attractor[0] == '0':
                            o_output_signal.d_out_value_to_attractor[0].append(o_attractor)
                        elif l_signals_in_attractor[0] == '1':
                            o_output_signal.d_out_value_to_attractor[1].append(o_attractor)
                if len(set(l_signals_in_local_scene)) == 1:
                    l_signals_for_output.append(l_signals_in_local_scene[0])
                else:
                    if len(set(l_signals_in_local_scene)) == 2:
                        l_signals_for_output.extend(l_signals_in_local_scene)
            if len(set(l_signals_for_output)) == 1:
                o_output_signal.kind_signal = 1
                print("INFO: ", "the output signal is restricted")
            elif len(set(l_signals_for_output)) == 2:
                o_output_signal.kind_signal = 3
                print("INFO: ", "the output signal is stable")
            else:
                o_output_signal.kind_signal = 4
                print("INFO: ", "the scene signal is not stable. This CBN dont have stable Attractor Fields")

    def find_compatible_pairs(self):
        """
        generate the pairs using the output signal
        :return: list of pair attractors by coupling signal
        """

        CustomText.make_title('FIND COMPATIBLE ATTRACTOR PAIRS')

        # count the number of pairs
        n_pairs = 0

        # for every local network finds compatible attractor pairs
        for o_local_network in self.l_local_networks:
            # find the output edges from the local network
            l_output_edges = self.get_output_edges_by_network_index(o_local_network.index)
            # find the pairs for every signal
            for o_output_signal in l_output_edges:
                # coupling the attractors pairs by the output signal 0
                l_attractors_input_0 = []
                for aux_attractor in o_output_signal.d_out_value_to_attractor[0]:
                    l_attractors_input_0.append(aux_attractor.g_index)

                # coupling the attractors pairs by the output signal 1
                l_attractors_input_1 = []
                for aux_attractor in o_output_signal.d_out_value_to_attractor[1]:
                    l_attractors_input_1.append(aux_attractor.g_index)

                l_pairs_edge_0 = []
                l_pairs_edge_1 = []

                # search the values for every signal
                for signal_value in o_output_signal.d_out_value_to_attractor.keys():
                    # find the attractors that generated by this signal
                    l_attractors_output = []
                    # select the attractor who generated the output value of the signal
                    for o_attractor in self.get_attractors_by_input_signal_value(o_output_signal.index_variable,
                                                                                 signal_value):
                        l_attractors_output.append(o_attractor.g_index)
                        # o_attractor.show()
                    if signal_value == 0:
                        l_pairs_edge_0 = list(itertools.product(l_attractors_input_0, l_attractors_output))
                    elif signal_value == 1:
                        l_pairs_edge_1 = list(itertools.product(l_attractors_input_1, l_attractors_output))
                # Join the two list in only one
                o_output_signal.d_comp_pairs_attractors_by_value[0] = l_pairs_edge_0
                o_output_signal.d_comp_pairs_attractors_by_value[1] = l_pairs_edge_1

                n_pairs += len(o_output_signal.d_comp_pairs_attractors_by_value[0])
                n_pairs += len(o_output_signal.d_comp_pairs_attractors_by_value[1])

        print('Number of attractor pairs:', n_pairs)
        CustomText.make_sub_sub_title('END FIND ATTRACTOR PAIRS')

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

    def mount_stable_attractor_fields(self):
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

            # Extract the index local network from each attractor pair.
            base_attractor_pairs = [attractor for pair in base_pairs for attractor in pair]

            # generate the list of already networks visited
            l_already_networks = []
            for g_i_attractor in base_attractor_pairs:
                l_already_networks.append(self.d_local_attractors[g_i_attractor][0])
            l_already_networks = set(l_already_networks)

            # Check if any local network from the candidate attractor pair is present in the base
            double_check = 0
            for candidate_attractor_g_index in candidate_pair:
                if self.d_local_attractors[candidate_attractor_g_index][0] in l_already_networks:
                    if candidate_attractor_g_index in base_attractor_pairs:
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

        CustomText.make_title('FIND ATTRACTOR FIELDS')

        # # Create a Dictionary only with the important information
        # self.generate_attractor_dictionary()

        # Order the edges by compatibility
        self.order_edges_by_compatibility()

        # generate the base list of pairs made with the pairs made with 0 or 1 coupĺing signal
        l_base_pairs = (self.l_directed_edges[0].d_comp_pairs_attractors_by_value[0]
                        + self.l_directed_edges[0].d_comp_pairs_attractors_by_value[1])

        # for every edge make the union to the base
        for o_directed_edge in self.l_directed_edges[1:]:
            l_candidate_pairs = o_directed_edge.d_comp_pairs_attractors_by_value[0] + \
                                o_directed_edge.d_comp_pairs_attractors_by_value[1]
            # join the base list with the new directed edge
            l_base_pairs = cartesian_product_mod(l_base_pairs, l_candidate_pairs)
            # print(l_base_pairs)

            # If the base of pairs don't have elements, break the for and ends the algorithm ends
            if len(l_base_pairs) == 0:
                break

        # generate a dictionary of the attractor fields
        self.d_attractor_fields = {}
        attractor_field_count = 1
        for base_element in l_base_pairs:
            self.d_attractor_fields[attractor_field_count] = list(set(item for pair in base_element for item in pair))
            attractor_field_count += 1

        print("Number of attractor fields found:", len(l_base_pairs))
        CustomText.make_sub_sub_title("END MOUNT ATTRACTOR FIELDS")

    # SHOW FUNCTIONS

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
                # print("RESTRICTED SIGNAL")
        print("Number of restricted signals :", n_restricted_signals)

    def show_description(self):
        CustomText.make_title('CBN description')
        l_local_networks_indexes = [o_local_network.index for o_local_network in self.l_local_networks]
        CustomText.make_sub_title(f"Local Networks: {l_local_networks_indexes}")
        for o_local_network in self.l_local_networks:
            o_local_network.show()
        CustomText.make_sub_title(f"Directed edges: {l_local_networks_indexes}")
        # for o_directed_edge in self.l_directed_edges:
        #     o_directed_edge.show()

    def show_global_scenes(self):
        CustomText.make_sub_title('LIST OF GLOBAL SCENES')
        for o_global_scene in self.l_global_scenes:
            o_global_scene.show()

    def show_local_attractors(self):
        CustomText.make_title('Show local attractors')
        for o_local_network in self.l_local_networks:
            CustomText.make_sub_title(f"Network {o_local_network.index}")
            for o_scene in o_local_network.l_local_scenes:
                CustomText.make_sub_sub_title(f"Network: {o_local_network.index} " +
                                              f"- Scene: {o_scene.l_values} " +
                                              f"- N. of Attractors: {len(o_scene.l_attractors)}")
                print("Network:", o_local_network.index, "- Scene:", o_scene.l_values)
                print("Attractors number:", len(o_scene.l_attractors))
                for o_attractor in o_scene.l_attractors:
                    CustomText.print_simple_line()
                    print(f"Global index: {o_attractor.g_index} -> {self.d_local_attractors[o_attractor.g_index]}")
                    for o_state in o_attractor.l_states:
                        print(o_state.l_variable_values)

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
                    print(o_pair)

    def show_stable_attractor_fields(self):
        CustomText.print_duplex_line()
        print("Show the list of attractor fields")
        print("Number Stable Attractor Fields:", len(self.d_attractor_fields))
        for key, o_attractor_field in self.d_attractor_fields.items():
            CustomText.print_simple_line()
            print(key)
            print(o_attractor_field)

    def show_resume(self):
        CustomText.make_title('CBN Detailed Resume')
        CustomText.make_sub_sub_title('Principal characteristics')
        CustomText.print_simple_line()
        print('Number of local networks:', len(self.l_local_networks))
        print('Number of variables per local network:', self.get_n_local_variables())
        print('Kind Topology:', self.get_kind_topology())
        print('Number of input variables:', self.get_n_input_variables())
        print('Number of output variables:', self.get_n_output_variables())

        CustomText.make_sub_sub_title("Indicators")
        CustomText.print_simple_line()
        print("Number of local attractors:", self.get_n_local_attractors())
        print("Number of attractor pairs:", self.get_n_pair_attractors())
        print("Number of attractor fields:", self.get_n_attractor_fields())
        CustomText.print_simple_line()

    # GET FUNCTIONS
    def get_network_by_index(self, index):
        for o_local_network in self.l_local_networks:
            if o_local_network.l_index == index:
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
        return len(self.d_attractor_fields)

    def get_n_local_variables(self):
        return len(self.l_local_networks[0].l_var_intern)

    def get_kind_topology(self):
        pass

    def get_n_input_variables(self):
        pass

    def get_n_output_variables(self):
        pass

    def generate_attractor_dictionary(self):
        """
        Generates a Dictionary of local attractors
        :return: a list of triples (a,b,c) where:
         - 'a' is the network index
         - 'b' is the scene index
         - 'c' is the local attractor index
        """
        d_local_attractors = {}
        for o_local_network in self.l_local_networks:
            for o_scene in o_local_network.l_local_scenes:
                for o_attractor in o_scene.l_attractors:
                    t_triple = (o_local_network.index, o_scene.index, o_attractor.l_index)
                    d_local_attractors[o_attractor.g_index] = t_triple

        self.d_local_attractors = d_local_attractors

    def get_local_attractor_by_index(self, i_attractor):
        for o_local_network in self.l_local_networks:
            for o_scene in o_local_network.l_local_scenes:
                for o_attractor in o_scene.l_attractors:
                    if o_attractor.g_index == i_attractor:
                        return o_attractor
        print('ERROR: Attractor index not found')
        return None

    def show_local_attractors_dictionary(self):
        CustomText.make_title('Global Dictionary of local attractors')
        for key, value in self.d_local_attractors.items():
            print(key, '->', value)

    def show_stable_attractor_fields_detailed(self):
        CustomText.print_duplex_line()
        print("Show the list of attractor fields")
        print("Number Stable Attractor Fields:", len(self.d_attractor_fields))
        for key, value in self.d_attractor_fields.items():
            CustomText.print_simple_line()
            print(key, '->', value)
            for i_attractor in value:
                print(i_attractor, '->', self.d_local_attractors[i_attractor])
                # get and show the local attractor
                o_attractor = self.get_local_attractor_by_index(i_attractor)
                o_attractor.show()

    def show_attractors_fields(self):
        CustomText.make_sub_title('List of attractor fields')
        for key, value in self.d_attractor_fields.items():
            print(key, "->", value)
        print(f"Number of attractor fields found: {len(self.d_attractor_fields)}")

    def generate_global_scenes(self):
        CustomText.make_title('Generated Global Scenes')
        l_edges_indexes = []
        for o_directed_edge in self.l_directed_edges:
            l_edges_indexes.append(o_directed_edge.index_variable)

        binary_combinations = list(product([0, 1], repeat=len(l_edges_indexes)))
        self.l_global_scenes = []

        for combination in binary_combinations:
            o_global_scene = GlobalScene(l_edges_indexes, list(combination))
            self.l_global_scenes.append(o_global_scene)
        CustomText.make_sub_title('Global Scenes generated')

    def plot_topology(self, ax=None):
        self.o_global_topology.plot_topology(ax=ax)

    def count_fields_by_global_scenes(self):
        CustomText.make_sub_title('Counting the stable attractor fields by global scene')
        # Dictionary to store combinations and their counts
        d_global_scenes_count = {}

        for key, o_attractor_field in self.d_attractor_fields.items():
            # print(key, " : ", o_attractor_field)
            # Dictionary to store variable values
            d_variable_value = {}

            # Search the scenario of the attractor fields
            for i_attractor in o_attractor_field:
                o_attractor = self.get_local_attractor_by_index(i_attractor)
                for aux_pos, aux_variable in enumerate(o_attractor.relation_index):
                    d_variable_value[aux_variable] = o_attractor.local_scene[aux_pos]

            # Sort the dictionary by key
            sorted_dict = {k: d_variable_value[k] for k in sorted(d_variable_value)}
            # print(sorted_dict)

            # Create the key by concatenating the sorted values
            combination_key = ''.join(str(sorted_dict[k]) for k in sorted(sorted_dict))

            # Update the count of the combination in the dictionary
            if combination_key in d_global_scenes_count:
                d_global_scenes_count[combination_key] += 1
            else:
                d_global_scenes_count[combination_key] = 1

        # Print the dictionary of combinations and their counts
        print(d_global_scenes_count)

    @staticmethod
    def cbn_generator(v_topology, n_local_networks, n_vars_network, n_input_variables, n_output_variables,
                      n_max_of_clauses=None, n_max_of_literals=None, n_edges=None, base_graph=None):

        """
        Generate a special CBN

        Args:
            n_output_variables:
            n_input_variables:
            n_vars_network:
            n_max_of_literals:
            n_max_of_clauses:
            v_topology: The topology of the CBN cam be 'aleatory'
            n_local_networks: The number of local networks
            n_edges: the number of edges between local networks
            base_graph: a graph who is base for the new graph
        Returns:
            A CBN generated from a template

        """

        # GENERATE THE GLOBAL TOPOLOGY
        o_global_topology = GlobalTopology.generate_sample_topology(v_topology=v_topology, n_nodes=n_local_networks,
                                                                    n_edges=n_edges)

        # GENERATE THE LOCAL NETWORK TEMPLATE
        o_template = LocalNetworkTemplate(v_topology=v_topology,
                                          n_vars_network=n_vars_network,
                                          n_input_variables=n_input_variables,
                                          n_output_variables=n_output_variables,
                                          n_max_of_clauses=n_max_of_clauses,
                                          n_max_of_literals=n_max_of_literals)

        # GENERATE THE CBN WITH THE TOPOLOGY AND TEMPLATE
        o_cbn = CBN.generate_cbn_from_template(v_topology=v_topology,
                                               n_local_networks=n_local_networks,
                                               n_vars_network=n_vars_network,
                                               o_template=o_template,
                                               l_global_edges=o_global_topology.l_edges)

        return o_cbn

    @staticmethod
    def generate_cbn_from_template(v_topology, n_local_networks, n_vars_network, o_template, l_global_edges):

        # generate the local networks with the indexes and variables (without relations or dynamics)
        l_local_networks = CBN.generate_local_networks_indexes_variables(n_local_networks=n_local_networks,
                                                                         n_vars_network=n_vars_network)

        # generate the directed edges between the local networks
        l_directed_edges = []

        # Get the last index of the variables for the indexes of the directed edges
        i_last_variable = l_local_networks[-1].l_var_intern[-1] + 1

        # generate the directed edges given the last variable generated and the selected output variables
        i_directed_edge = 1
        for relation in l_global_edges:
            output_local_network = relation[0]
            input_local_network = relation[1]

            # get the output variables from template
            l_output_variables = o_template.get_output_variables_from_template(output_local_network, l_local_networks)

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
        l_local_networks = CBN.generate_local_dynamic_with_template(o_template=o_template,
                                                                    l_local_networks=l_local_networks,
                                                                    l_directed_edges=l_directed_edges)

        # generate the special coupled boolean network
        o_cbn = CBN(l_local_networks=l_local_networks,
                    l_directed_edges=l_directed_edges)

        # add the Global Topology Object
        o_global_topology = GlobalTopology(v_topology=v_topology, l_edges=l_global_edges)
        o_cbn.o_global_topology = o_global_topology

        return o_cbn

    @staticmethod
    def generate_local_dynamic_with_template(o_template, l_local_networks, l_directed_edges):
        """
        GENERATE THE DYNAMICS OF EACH LOCAL NETWORK
        :param o_template:
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

            # # find the directed edges by network index
            # l_input_signals_by_network = CBN.find_input_edges_by_network_index(index=o_local_network.index,
            #                                                                    l_directed_edges=l_directed_edges)

            # add the variable index of the directed edges
            # for o_signal in l_input_signals_by_network:
            #     o_local_network.l_var_exterm.append(o_signal.index_variable)
            # o_local_network.l_var_total = o_local_network.l_var_intern + o_local_network.l_var_exterm

            # generate the function description of the variables
            des_funct_variables = []
            # generate clauses for every local network adapting the template
            for i_local_variable in o_local_network.l_var_intern:
                CustomText.print_simple_line()
                # adapting the clause template to the specific variable
                l_clauses_node = CBN.update_clause_from_template(o_template=o_template,
                                                                 l_local_networks=l_local_networks,
                                                                 o_local_network=o_local_network,
                                                                 i_local_variable=i_local_variable,
                                                                 l_directed_edges=l_directed_edges)

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
    def update_clause_from_template(o_template, l_local_networks, o_local_network,
                                    i_local_variable, l_directed_edges):
        """
        update clause from template
        :param o_template
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
        pre_l_clauses_node = o_template.d_variable_cnf_function[i_template_variable]

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
