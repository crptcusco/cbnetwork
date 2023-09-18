from itertools import product  # generate the permutations
from random import randint  # generate random numbers integers
from matplotlib import pyplot as plt  # generate the figures
import random  # generate random numbers
import networkx as nx  # generate networks

from classes.internalvariable import InternalVariable
from classes.localnetwork import LocalNetwork
from classes.couplingsignal import CouplingSignal
from classes.utils.heap import Node, CustomHeap


class DirectedEdge:
    def __init__(self, index, input_network, output_network):
        self.index = index
        self.input_network = input_network
        self.output_network = output_network


class CBN:
    def __init__(self, l_local_networks, l_directed_edges):
        self.l_local_networks = l_local_networks
        self.l_directed_edges = l_directed_edges

        #DEJAR LAS SENHALES DE ACOPLAMIENTO AFUERA de la red local??? EN CBN????

        # Calculated properties
        self.l_attractor_fields = []

    def show(self):
        print("CBN description")
        l_local_networks_indexes = [o_local_network.index for o_local_network in self.l_local_networks]
        print("Local Networks:", l_local_networks_indexes)
        print("Directed edges:")
        for o_relation in self.l_directed_edges:
            o_relation.show()
            
    def show_attractors_fields(self):
        pass

    def process_output_signals(self):
        # update output signals for every local network
        for o_local_network in self.l_local_networks:
            for t_relation in self.l_directed_edges:
                if o_local_network.index == t_relation[1]:
                    o_local_network.l_output_signals.append(t_relation)
                    print(t_relation)

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

    @staticmethod
    def generate_aleatory_cbn(n_local_networks, n_var_network, n_relations, n_output_variables, n_clauses_function,
                              relations_fixed=False):
        print("Generating the CBN")
        print("==================")
        # GENERATE THE LOCAL NETWORKS IN BASIC FORM (WITHOUT RELATIONS AND DYNAMIC)
        l_local_networks = []
        l_directed_edges = []
        v_cont_var = 1
        for v_num_network in range(1, n_local_networks + 1):
            # generate a local network
            l_var_intern = list(range(v_cont_var, v_cont_var + n_var_network))
            o_local_network = LocalNetwork(v_num_network, l_var_intern)
            l_local_networks.append(o_local_network)
            v_cont_var = v_cont_var + n_var_network

        # GENERATE COUPLING SIGNALS IN ONE AUXILIARY LIST
        aux1_l_local_networks = []
        for o_local_network in l_local_networks:
            if relations_fixed:
                n_signals_local_network = n_relations
            else:
                n_signals_local_network = randint(1, n_relations)

            l_aux_local_networks = l_local_networks.copy()
            l_aux_local_networks.remove(o_local_network)
            l_local_networks_co = random.sample(l_aux_local_networks, n_signals_local_network)
            l_signals = []
            for o_local_network_co in l_local_networks_co:
                l_output_variables = random.sample(o_local_network_co.l_var_intern, n_output_variables)
                if n_output_variables == 1:
                    coupling_function = l_output_variables[0]
                else:
                    coupling_function = " " + " ∨ ".join(list(map(str, l_output_variables))) + " "
                o_coupling_signal = CouplingSignal(o_local_network.index, o_local_network_co.index,
                                                   l_output_variables, v_cont_var, coupling_function)
                l_signals.append(o_coupling_signal)
                o_directed_edge = DirectedEdge(v_cont_var, o_local_network.index, o_local_network_co.index)
                l_directed_edges.append(o_directed_edge)
                v_cont_var = v_cont_var + 1
            o_local_network.l_input_signals = l_signals.copy()

            aux1_l_local_networks.append(o_local_network)
        l_local_networks = aux1_l_local_networks.copy()

        # GENERATE THE DYNAMICS OF EACH RDD
        number_max_of_clauses = n_clauses_function
        number_max_of_literals = 3
        # we generate an auxiliary list to add the coupling signals
        aux2_l_local_networks = []
        for o_local_network in l_local_networks:
            # Create a list of all RDDAs variables
            l_aux_variables = []
            # Add the variables of the coupling signals
            for o_signal in o_local_network.l_input_signals:
                l_aux_variables.append(o_signal.index_variable_signal)
            # add local variables
            l_aux_variables.extend(o_local_network.l_var_intern)

            # generate the function description of the variables
            des_funct_variables = []
            # generate clauses
            for v_description_variable in o_local_network.l_var_intern:
                l_clauses_node = []
                for v_clause in range(0, randint(1, number_max_of_clauses)):
                    v_num_variable = randint(1, number_max_of_literals)
                    # randomly select from the signal variables
                    l_literals_variables = random.sample(l_aux_variables, v_num_variable)
                    l_clauses_node.append(l_literals_variables)
                # adding the description of variable in form of object
                o_variable_model = InternalVariable(v_description_variable, l_clauses_node)
                des_funct_variables.append(o_variable_model)
                # adding the description in functions of every variable
            # adding the local network to list of local networks
            o_local_network.des_funct_variables = des_funct_variables.copy()
            aux2_l_local_networks.append(o_local_network)
            # actualized the list of local networks
        l_local_networks = aux2_l_local_networks.copy()

        # Process the input signals
        for o_local_network in l_local_networks:
            o_local_network.process_input_signals()
        # Process the output signals
        for o_local_network in l_local_networks:
            o_local_network.process_output_signals()
            o_local_network.show()
            print("Local network created")
            print("---------------------")

        o_cbn = CBN(l_local_networks, l_directed_edges)
        print("Coupled Boolean Network created")
        print("===============================")
        return o_cbn

    def find_attractors(self):
        print("Find Attractors using optimized method")
        print("-------------------------")
        print("Begin of the initial loop")

        # Defined the kind for every coupling signal: stable 1, not compute 2
        #     1: "restricted",
        #     2: "not compute",
        #     3: "stable",
        #     4: "not stable"

        # Assigning the king of the relations, all the relations are not computed
        for o_directed_edge in self.l_directed_edges:
            o_directed_edge.kind_relation = 2
            # print(o_relation.kind_relation)

        # calculate the initial weights for every node (local network)
        # create an empty heap
        o_custom_heap = CustomHeap()
        for o_local_network in self.l_local_networks:
            # initial graph only have not computed signals
            weight = 0
            for o_directed_edge in self.l_directed_edges:
                if o_directed_edge.input_network == o_local_network.index:
                    weight = weight + o_directed_edge.kind_relation
            # add node to the heap with computed weight
            o_node = Node(o_local_network.index, weight)
            o_custom_heap.add_node(o_node)

        # PROCESS THE FIRST NODE - FIND ATTRACTORS
        # find the node in the top  of the heap
        lowest_weight_node = CustomHeap.remove_node(o_custom_heap)
        # find the local network information
        o_local_network = self.find_network_by_index(lowest_weight_node.index)
        # calculate the local scenarios
        l_local_scenes = list(product(list('01'), repeat=len(o_local_network.l_input_signals)))
        # calculate the attractors for the node in the top of the  heap
        o_local_network = LocalNetwork.find_local_scenery_attractors(o_local_network, l_local_scenes)
        # update the network in the CBN
        self.update_network_by_index(lowest_weight_node.index, o_local_network)

        # validate if the output variables by attractor send a fixed value
        for o_local_scene in o_local_network.l_local_scenes:
            # print("Scene: ", str(o_local_scene.l_values))
            for o_output_signal in o_local_network.l_output_signals:
                print(o_output_signal.true_table())



        # for l_attractor_scene in l_scenery_attractors:
        #     print("Local scenery :", l_attractor_scene[0])
        #     print(l_attractor_scene[1])
        #     # evaluate the kind of coupling signal that the attractor send
        #     for o_signal in o_local_network.l_input_signals:
        #         print("hola")
        #         print(o_signal.l_output_variables)
        #         print(o_signal.index_variable_signal)
        #         print(o_signal.coupling_function)
        #         print(o_signal.true_table)

        # update the table of kinds of coupling signals
        # o_relation.kind_relation

        # update the weights of the nodes

        print("Update the weights of the nodes")
        # o_custom_heap = CustomHeap()
        # calculate the initial weights for every node
        for o_local_network_index in o_custom_heap.get_indexes():
            o_local_network = self.find_network_by_index(o_local_network_index)
            weight = 0
            # for o_signal in o_local_network.l_input_signals:

            # initial graph only have not computed signals
            weight = len(o_local_network.l_input_signals) * 2
            # add edge to the heap
            o_node = Node(o_local_network.index, weight)
            o_custom_heap.add_node(o_node)

        print("end of the initial loop")
        print("-----------------------")

        # Verify if the attractors are not compute
        while o_custom_heap.get_size() > 1:
            # calculate the weight of coupling signals
            # calculate the kind for every local network an every coupling signal

            # for o_signal in o_local_network.l_input_signals:
            #     print(o_signal.index_variable_signal)

            # Update the weight for every local network in the heap
            # local_weight = Node.calculate_weight(l_signal_status)

            lowest_weight_node = CustomHeap.remove_node(o_custom_heap)
            # Find Local Network
            o_local_network = self.find_network_by_index(lowest_weight_node.index)
            # Find attractors with the minimum weight
            LocalNetwork.find_local_scenery_attractors(o_local_network, l_local_scenes)
            print(lowest_weight_node.index, lowest_weight_node.weight)

            # Update the weights
            for o_node in o_custom_heap.heap:
                print(o_node)

        print("All the attractors are computed")
        print("===============================")

        # Remove a node from the heap

        # Update the weight of the nodes

        # Stop criteria
        # if weight  == 3 or weight = 0

        # weight = 0
        # kind_coupling_flag = False
        # while not kind_coupling_flag:

        # Crear un heap con el numero de sinais de acoplamento que tem while weigth == 0 or weigth  == 3 :
        # self.l_attractors = [] for o_local_network in self.l_local_networks: for o_input_signal in
        # o_local_network.l_input_signals: print("Add edge:", o_input_signal.local_network_output, "-",
        # o_input_signal.local_network_input, ':', 0) o_graph.add_edge(o_input_signal.local_network_output,
        # o_input_signal.local_network_input, weight=0)

        # @staticmethod
        # def calculate_weight(l_signals_status):
        #     # l_signals_status = [0, 0, 1, 2]
        #     res = sum(l_signals_status)
        #     return res

        # # Types Coupling Signals
        # kind_coupling_signals = {
        #     1: "not compute",
        #     2: "restricted",
        #     3: "stable",
        #     4: "not stable"
        # }

        # # How to compute the weight
        # dict_weight = {
        #     1: "stable",
        #     2: "not compute"
        # }

        # # Evaluate the signals that don't have input coupling signals
        # l_local_network_without_signals = []
        # for o_local_network in self.l_local_networks:
        #     if not o_local_network.l_input_signals:
        #         l_local_network_without_signals.append(o_local_network.index)
        # print(l_local_network_without_signals)

        # print(heap)

    def evaluate_cbn_topology(self):
        # Find attractors
        # create a directed graph
        o_graph = nx.DiGraph()

        # add edges to the graph
        for o_local_network in self.l_local_networks:
            for o_input_signal in o_local_network.l_input_signals:
                print("Add edge:", o_input_signal.local_network_output, "-", o_input_signal.local_network_input, ':', 0)
                o_graph.add_edge(o_input_signal.local_network_output, o_input_signal.local_network_input, weight=0)

        # graph have cycles or not
        is_acyclic = nx.is_directed_acyclic_graph(o_graph)
        if is_acyclic:
            # make topological order
            topological_order = list(nx.topological_sort(o_graph))
            print("The graph is no cycled - Topological order:", topological_order)
        else:
            print("The graph is cycled - you have to use other strategy ... using heaps")
