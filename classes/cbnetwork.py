import itertools
from itertools import product  # generate the permutations
from random import randint  # generate random numbers integers
from matplotlib import pyplot as plt  # generate the figures
import random  # generate random numbers
import networkx as nx  # generate networks

from classes.internalvariable import InternalVariable
from classes.localnetwork import LocalNetwork
from classes.directededge import DirectedEdge
from classes.utils.customheap import Node, CustomHeap


class CBN:
    def __init__(self, l_local_networks, l_directed_edges):

        self.l_local_networks = l_local_networks
        self.l_directed_edges = l_directed_edges

        # Calculated properties
        self.l_output_edges = None
        self.l_global_scenes = []
        self.l_attractor_fields = []

    def show_cbn(self):
        print("INFO:", "CBN description")
        l_local_networks_indexes = [o_local_network.index for o_local_network in self.l_local_networks]
        print("INFO:", "Local Networks:", l_local_networks_indexes)
        print("INFO:", "Directed edges:")
        for o_directed_edge in self.l_directed_edges:
            o_directed_edge.show()

    def show_attractors(self):
        for o_network in self.l_local_networks:
            print("==============")
            print("Network:", o_network.index)
            for o_scene in o_network.l_local_scenes:
                print("--------------")
                print("Network:", o_network.index, "- Scene:", o_scene.l_values)
                print("Attractors number:", len(o_scene.l_attractors))
                for o_attractor in o_scene.l_attractors:
                    print("--------------")
                    for o_state in o_attractor.l_states:
                        print(o_state.l_variable_values)

    def show_global_scenes(self):
        for o_global_scene in self.l_global_scenes:
            print("INFO:", "Global scene -", o_global_scene)

    def show_attractors_fields(self):
        pass

    @staticmethod
    def show_allowed_topologies():
        # allowed topologies
        allowed_topologies = {
            1: "complete_graph",
            2: "binomial_tree",
            3: "cycle_graph",
            4: "path_graph",
            5: "gn_graph",
            6: "gnc_graph"
        }
        for key, value in allowed_topologies.items():
            print("INFO:", key, "-", value)

    @staticmethod
    def generate_cbn_topology(l_networks, v_topology=6):
        # We create a graph beginning in 1
        n_nodes = len(l_networks)
        G = nx.DiGraph()
        # classical topologies
        if v_topology == 1:
            G = nx.complete_graph(n_nodes, nx.DiGraph())
        elif v_topology == 2:
            G = nx.binomial_tree(n_nodes, nx.DiGraph())
        elif v_topology == 3:
            G = nx.cycle_graph(n_nodes, nx.DiGraph())
        elif v_topology == 4:
            G = nx.path_graph(n_nodes, nx.DiGraph())
        # aleatory topologies
        elif v_topology == 5:
            G = nx.gn_graph(n_nodes)
        elif v_topology == 6:
            G = nx.gnc_graph(n_nodes)
        else:
            G = nx.complete_graph(n_nodes, nx.DiGraph())

        # Classical topologies
        # G = nx.balanced_tree(n_nodes, 1, nx.DiGraph())
        # G = nx.circulant_graph(n, [1, 2], nx.DiGraph())
        # G = nx.full_rary_tree(2, n, nx.DiGraph())

        # # Directed Graphs
        # G = nx.gn_graph(n)
        # G = nx.gnr_graph(n,0.5) # need probabilities
        # G = nx.gnc_graph(n)
        # G = nx.random_k_out_graph(n) # not supported
        # G = nx.scale_free_graph(n) # have cycles

        # Renaming the label of the nodes for beginning in 1
        mapping = {node: node + 1 for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping)
        return list(G.edges)

    @staticmethod
    def generate_cbn(n_local_networks, n_var_network, v_topology, n_output_variables, n_clauses_function):
        print("MESSAGE:", "Generating the CBN")
        print("==================")
        # GENERATE THE LOCAL NETWORKS IN BASIC FORM (WITHOUT RELATIONS AND DYNAMIC)
        l_local_networks = []
        l_directed_edges = []
        v_cont_var = 1
        # generate the local networks
        for v_num_network in range(1, n_local_networks + 1):
            # generate the variables of the networks
            l_var_intern = list(range(v_cont_var, v_cont_var + n_var_network))
            o_local_network = LocalNetwork(v_num_network, l_var_intern)
            l_local_networks.append(o_local_network)
            v_cont_var = v_cont_var + n_var_network

        # GENERATE THE TOPOLOGY
        l_relations = CBN.generate_cbn_topology(l_local_networks, v_topology)
        aux1_l_local_networks = []
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
                o_directed_edge = DirectedEdge(o_local_network.index, o_local_network_co.index,
                                               l_output_variables, v_cont_var, coupling_function)
                l_directed_edges.append(o_directed_edge)
                v_cont_var = v_cont_var + 1
            aux1_l_local_networks.append(o_local_network)
        l_local_networks = aux1_l_local_networks.copy()

        # Process the input and output signals for local_network
        for o_local_network in l_local_networks:
            l_input_signals = DirectedEdge.find_input_edges_by_network_index(o_local_network.index, l_directed_edges)
            o_local_network.process_input_signals(l_input_signals)
            # l_output_signals = DirectedEdge.find_output_edges_by_network_index(o_local_network.index,
            # l_directed_edges)
            # o_local_network.process_output_signals(l_input_signals)

        # GENERATE THE DYNAMICS OF EACH RDD
        number_max_of_clauses = n_clauses_function
        number_max_of_literals = 3
        # we generate an auxiliary list to add the coupling signals
        aux2_l_local_networks = []
        for o_local_network in l_local_networks:
            # Create a list of all RDDAs variables
            l_aux_variables = []
            # Add the variables of the coupling signals
            l_input_signals = DirectedEdge.find_input_edges_by_network_index(o_local_network.index, l_directed_edges)
            for o_signal in l_input_signals:
                l_aux_variables.append(o_signal.index_variable)
            # add local variables
            l_aux_variables.extend(o_local_network.l_var_intern)

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
                # adding the description of variable in form of object
                o_variable_model = InternalVariable(i_local_variable, l_clauses_node)
                des_funct_variables.append(o_variable_model)
                # adding the description in functions of every variable
            # adding the local network to list of local networks
            o_local_network.des_funct_variables = des_funct_variables.copy()
            aux2_l_local_networks.append(o_local_network)
            print("MESSAGE:", "Local network created :", o_local_network.index)
            print("---------------------")
            # actualized the list of local networks
        l_local_networks = aux2_l_local_networks.copy()

        o_cbn = CBN(l_local_networks, l_directed_edges)
        print("MESSAGE:", "Coupled Boolean Network created")
        print("===============================")
        return o_cbn

    def generate_global_scenes(self):
        print("MESSAGE:", "GENERATE GLOBAL SCENES")
        # generate the global scenes using all the combinations
        self.l_global_scenes = list(product(list('01'), repeat=len(self.l_directed_edges)))
        # for global_scene in self.l_global_scenes:
        #     print("INFO:", global_scene)
        # print(self.l_global_scenes)
        print("MESSAGE:", "Global Scenes generated")
        print("==================================")

    def process_output_signals(self):
        # update output signals for every local network
        for o_local_network in self.l_local_networks:
            for t_relation in self.l_directed_edges:
                if o_local_network.index == t_relation[1]:
                    o_local_network.l_output_signals.append(t_relation)
                    print("INFO:", t_relation)

    def find_network_by_index(self, index):
        for o_local_network in self.l_local_networks:
            if o_local_network.index == index:
                return o_local_network

    def update_network_by_index(self, index, o_local_network_update):
        for o_local_network in self.l_local_networks:
            if o_local_network.index == index:
                o_local_network = o_local_network_update
                print("MESSAGE:", "Local Network updated")
                return True
        print("ERROR:", "Local Network not found")
        return False

    def generate_graph(self):
        G = nx.DiGraph()
        l_networks = []
        for o_edge in self.l_directed_edges:
            l_networks.append((o_edge.input_local_network, o_edge.output_local_network))
        G.add_edges_from(l_networks)
        nx.draw(G)

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

    def find_attractors(self):
        print("MESSAGE:", "Find Attractors using optimized method")
        print("-------------------------")
        print("MESSAGE:", "Begin of the initial loop")

        # In the beginning all the kind or relations are "not computed" with index 2
        # print("INFO:",o_directed_edge.index_variable_signal, ":", o_directed_edge.kind_relation)

        # create an empty heap to organize the local networks by weight
        o_custom_heap = CustomHeap()
        # calculate the initial weights for every node (local network)
        for o_local_network in self.l_local_networks:
            # initial graph only have not computed signals
            weight = 0
            for o_directed_edge in self.l_directed_edges:
                if o_directed_edge.input_local_network == o_local_network.index:
                    weight = weight + o_directed_edge.kind_signal
            # add node to the heap with computed weight
            o_node = Node(o_local_network.index, weight)
            o_custom_heap.add_node(o_node)

        # print("INITIAL HEAP")
        initial_heap = o_custom_heap.get_indexes()
        # print(initial_heap)

        # PROCESS THE FIRST NODE - FIND ATTRACTORS
        # find the node in the top  of the heap
        lowest_weight_node = CustomHeap.remove_node(o_custom_heap)
        # find the local network information
        o_local_network = self.find_network_by_index(lowest_weight_node.index)
        # calculate the local scenarios
        l_local_scenes = None
        if len(o_local_network.l_var_exterm) != 0:
            l_local_scenes = list(product(list('01'), repeat=len(o_local_network.l_var_exterm)))

        # calculate the attractors for the node in the top of the  heap
        o_local_network = LocalNetwork.find_local_attractors(o_local_network, l_local_scenes)
        # # update the network in the CBN
        # self.update_network_by_index(lowest_weight_node.index, o_local_network)

        # # Update kind signals
        # validate if the output variables by attractor send a fixed value
        l_directed_edges = DirectedEdge.find_output_edges_by_network_index(o_local_network.index, self.l_directed_edges)
        # print("INFO:", "Local network:", o_local_network.index)
        for o_output_signal in l_directed_edges:
            # print("INFO:", "Index variable output signal:", o_output_signal.index_variable_signal)
            # print("INFO:", "Output variables:", o_output_signal.l_output_variables)
            # print("INFO:", str(o_output_signal.true_table))
            l_signals_for_output = []
            for o_local_scene in o_local_network.l_local_scenes:
                # print("INFO:", "Scene: ", str(o_local_scene.l_values))
                l_signals_in_local_scene = []
                for o_attractor in o_local_scene.l_attractors:
                    # print("INFO:", "ATTRACTOR")
                    l_signals_in_attractor = []
                    for o_state in o_attractor.l_states:
                        # print("INFO:", "STATE")
                        # print("INFO:", o_local_network.l_var_total)
                        # print("INFO:", o_local_network.l_var_intern)
                        # print("INFO:", o_state.l_variable_values)
                        # # select the values of the output variables
                        true_table_index = ""
                        for v_output_variable in o_output_signal.l_output_variables:
                            # print("INFO:", "Variables list:", o_local_network.l_var_total)
                            # print("INFO:", "Output variables list:", o_output_signal.l_output_variables)
                            # print("INFO:", "Output variable:", v_output_variable)
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
                        print("MESSAGE:", "the attractor signal value is stable")

                        # add the attractor to the dictionary of output value -> attractors
                        if l_signals_in_attractor[0] == '0':
                            o_output_signal.d_out_value_to_attractor[0].append(o_attractor)
                        elif l_signals_in_attractor[0] == '1':
                            o_output_signal.d_out_value_to_attractor[1].append(o_attractor)

                    else:
                        print("MESSAGE:", "the attractor signal is not stable")
                if len(set(l_signals_in_local_scene)) == 1:
                    l_signals_for_output.append(l_signals_in_local_scene[0])
                    print("MESSAGE:", "the scene signal is restricted")
                else:
                    if len(set(l_signals_in_local_scene)) == 2:
                        l_signals_for_output.extend(l_signals_in_local_scene)
                        print("MESSAGE:", "the scene signal value is stable")
                    else:
                        print("warning:", "the scene signal is not stable")
            if len(set(l_signals_for_output)) == 1:
                o_output_signal.kind_signal = 1
                print("MESSAGE:", "the output signal is restricted")
            elif len(set(l_signals_for_output)) == 2:
                o_output_signal.kind_signal = 3
                print("MESSAGE:", "the output signal is stable")
            else:
                o_output_signal.kind_signal = 4
                print("error:", "the scene signal is not stable. This CBN dont have stable Attractor Fields")

        # # print all the kinds of the signals
        # print("MESSAGE:", "Resume")
        # print("Network:", o_local_network.index)
        # for o_directed_edge in self.l_directed_edges:
        #     print(o_directed_edge.index_variable_signal, ":", o_directed_edge.kind_signal)

        # Update the weights of the nodes
        # Add the output network to the list of modified networks
        l_modified_edges = DirectedEdge.find_input_edges_by_network_index(o_local_network.index, self.l_directed_edges)
        for o_edge in l_modified_edges:
            modified_network_index = o_edge.output_local_network
            print("INFO:", "Network", modified_network_index)
            print("INFO:", "Relation:", o_edge.input_local_network, "->", o_edge.output_local_network)
            weight = 0
            l_edges = DirectedEdge.find_input_edges_by_network_index(o_edge.output_local_network, self.l_directed_edges)
            for o_updated_edge in l_edges:
                weight = weight + o_updated_edge.kind_signal
            print("INFO:", "New weight:", weight)
            o_custom_heap.update_node(o_edge.output_local_network, weight)

        print("MESSAGE:", "INITIAL HEAP")
        print(initial_heap)
        print("MESSAGE:", "UPDATE HEAP")
        print(o_custom_heap.get_indexes())

        # Verify if the heap have at least two elements
        while o_custom_heap.get_size() > 0:
            # find the node on the top of the heap
            lowest_weight_node = CustomHeap.remove_node(o_custom_heap)
            # Find Local Network
            o_local_network = self.find_network_by_index(lowest_weight_node.index)

            l_local_scenes = None
            if len(o_local_network.l_var_exterm) != 0:
                l_local_scenes = list(product(list('01'), repeat=len(o_local_network.l_var_exterm)))

            # Find attractors with the minimum weight
            LocalNetwork.find_local_attractors(o_local_network, l_local_scenes)
            print("INFO:", "Local Network:", lowest_weight_node.index, "Weight:", lowest_weight_node.weight)

            # COPIED CODE !!!
            # # Update kind signals
            # validate if the output variables by attractor send a fixed value
            l_directed_edges = DirectedEdge.find_output_edges_by_network_index(o_local_network.index,
                                                                               self.l_directed_edges)
            print("INFO:", "Local network:", o_local_network.index)
            for o_output_signal in l_directed_edges:
                print("INFO:", "Index variable output signal:", o_output_signal.index_variable)
                print("INFO:", "Output variables:", o_output_signal.l_output_variables)
                print(str(o_output_signal.true_table))
                l_signals_for_output = []
                for o_local_scene in o_local_network.l_local_scenes:
                    print("INFO:", "Scene: ", str(o_local_scene.l_values))
                    l_signals_in_local_scene = []
                    for o_attractor in o_local_scene.l_attractors:
                        print("INFO:", "ATTRACTOR")
                        l_signals_in_attractor = []
                        for o_state in o_attractor.l_states:
                            print("INFO:", "STATE")
                            print("INFO:", o_local_network.l_var_total)
                            print("INFO:", o_local_network.l_var_intern)
                            print("INFO:", o_state.l_variable_values)
                            # select the values of the output variables
                            true_table_index = ""
                            for v_output_variable in o_output_signal.l_output_variables:
                                print("INFO:", "Variables list:", o_local_network.l_var_total)
                                print("INFO:", "Output variables list:", o_output_signal.l_output_variables)
                                print("INFO:", "Output variable:", v_output_variable)
                                pos = o_local_network.l_var_total.index(v_output_variable)
                                value = o_state.l_variable_values[pos]
                                true_table_index = true_table_index + str(value)
                            print("INFO:", o_output_signal.l_output_variables)
                            print("INFO:", true_table_index)
                            output_value_state = o_output_signal.true_table[true_table_index]
                            print("INFO:", "Output value :", output_value_state)
                            l_signals_in_attractor.append(output_value_state)
                        if len(set(l_signals_in_attractor)) == 1:
                            l_signals_in_local_scene.append(l_signals_in_attractor[0])
                            print("MESSAGE:", "the attractor signal value is stable")

                            # add the attractor to the dictionary of output value -> attractors
                            if l_signals_in_attractor[0] == '0':
                                o_output_signal.d_out_value_to_attractor[0].append(o_attractor)
                            elif l_signals_in_attractor[0] == '1':
                                o_output_signal.d_out_value_to_attractor[1].append(o_attractor)

                        else:
                            print("MESSAGE:", "the attractor signal is not stable")
                    if len(set(l_signals_in_local_scene)) == 1:
                        l_signals_for_output.append(l_signals_in_local_scene[0])
                        print("MESSAGE:", "the scene signal is restricted")
                    else:
                        if len(set(l_signals_in_local_scene)) == 2:
                            l_signals_for_output.extend(l_signals_in_local_scene)
                            print("MESSAGE:", "the scene signal value is stable")
                        else:
                            print("MESSAGE:", "the scene signal is not stable")
                if len(set(l_signals_for_output)) == 1:
                    o_output_signal.kind_signal = 1
                    print("MESSAGE:", "the output signal is restricted")
                elif len(set(l_signals_for_output)) == 2:
                    o_output_signal.kind_signal = 3
                    print("MESSAGE:", "the output signal is stable")
                else:
                    o_output_signal.kind_signal = 4
                    print("MESSAGE:", "THE SCENE SIGNAL IS NOT STABLE. THIS CBN DONT HAVE STABLE ATTRACTOR FIELDS")

            # print all the kinds of the signals
            print("===============================")
            print("INFO:", "RESUME")
            # print("INFO:", "Network:", o_local_network.index)
            # for o_directed_edge in self.l_directed_edges:
            #     print("INFO:", o_directed_edge.index_variable_signal, ":", o_directed_edge.kind_signal)

            # Update the weights of the nodes
            # Add the output network to the list of modified networks
            l_modified_edges = DirectedEdge.find_input_edges_by_network_index(o_local_network.index,
                                                                              self.l_directed_edges)
            for o_edge in l_modified_edges:
                modified_network_index = o_edge.output_local_network
                print("INFO:", "Network", modified_network_index)
                print("INFO:", "Relation:", o_edge.input_local_network, "->", o_edge.output_local_network)
                weight = 0
                l_edges = DirectedEdge.find_input_edges_by_network_index(o_edge.output_local_network,
                                                                         self.l_directed_edges)
                for o_updated_edge in l_edges:
                    weight = weight + o_updated_edge.kind_signal
                print("INFO:", "New weight:", weight)
                o_custom_heap.update_node(o_edge.output_local_network, weight)

            # print("INITIAL HEAP")
            # print(initial_heap)
            # print("UPDATE HEAP")
            # print(o_custom_heap.get_indexes())
            # print("empty heap")
            print("MESSAGE:", "The Local attractors are computed")
        print("MESSAGE:", "END")
        print("=========================")

    def find_compatible_pairs(self):
        print("===============================")
        print("FIND COMPATIBLE ATTRACTOR PAIRS")

        # generate the pairs using the output signal
        l_pairs = []
        # for every local networks find compatible attractor pairs
        for o_local_network in self.l_local_networks:
            print("----------------------------------------")
            # find the output edges from the local network
            l_output_edges = self.get_output_edges_by_network_index(o_local_network.index)
            # find the pairs for every signal
            for o_output_signal in l_output_edges:
                print("-------------------------")
                print("INFO:", "Network -", o_local_network.index, ", Output Signal - ", o_output_signal.index_variable)
                # Show the dictionary of the attractors by value of output signal
                o_output_signal.show_dict_v_output_signal_attractor()
                # coupling the attractors pairs by the output signal
                print("INFO:", "Output Attractor List ")
                # search the values for every signal
                for input_signal_value in o_output_signal.d_out_value_to_attractor.keys():
                    # Select the attractor who generated the output value of the signal
                    l_attractors_input = o_output_signal.d_out_value_to_attractor[input_signal_value]
                    # find the attractors that generated by this signal
                    l_attractors_output = []
                    # analyzed every attractor
                    for o_attractor in self.get_attractors_by_input_signal(o_output_signal.index_variable):
                        # get the position of the signal
                        v_pos = o_attractor.relation_index.index(o_output_signal.index_variable)
                        if o_attractor.local_scene[v_pos] == input_signal_value:
                            print(v_pos)
                            print(o_attractor.local_scene[v_pos])
                            l_attractors_output.append(o_attractor)
                            o_attractor.show()

                    # compute the compatible pair attractors
                    l_pairs_edge = list(itertools.product(l_attractors_input, l_attractors_output))
                    # print("INFO:", l_pairs_edge)
                    o_output_signal.l_comp_pairs_attractors = l_pairs_edge

    def get_attractors_by_input_signal(self, index_variable_signal):
        for o_local_network in self.l_local_networks:
            for scene in o_local_network.l_local_scenes:
                # Validate se tem sinais ou nao
                if scene.l_values is not None:
                    if index_variable_signal in scene.l_index_signals:
                        return scene.l_attractors

    def show_attractor_pairs(self):
        print("====================================================")
        print("MESSAGE:", "List of the compatible attractor pairs")
        for o_directed_edge in self.l_directed_edges:
            print("====================================================")
            print("MESSAGE:", "Edge ", o_directed_edge.output_local_network, "->", o_directed_edge.input_local_network)
            for t_compatible_pair in o_directed_edge.l_comp_pairs_attractors:
                print("----------------------------------------------------")
                t_compatible_pair[0].show()
                t_compatible_pair[1].show()

