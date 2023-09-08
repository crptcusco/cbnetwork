from random import randint  # generate random numbers integers

from classes.internalvariable import InternalVariable
from classes.localnetwork import LocalNetwork
from classes.couplingsignal import CouplingSignal
import random  # generate random numbers
import networkx as nx


class CBN:
    def __init__(self, l_local_networks, l_coupling_signals):
        self.l_local_networks = l_local_networks
        self.l_coupling_signals = l_coupling_signals

    def show(self):
        pass

    @staticmethod
    def generate_aleatory_cbn(n_local_networks, n_var_network, n_relations, n_output_variables, n_clauses_function,
                              relations_fixed=False):
        # GENERATE THE LOCAL NETWORKS IN BASIC FORM (WITHOUT RELATIONS AND DYNAMIC)
        l_local_networks = []
        l_coupling_signals = []
        v_cont_var = 1
        for v_num_network in range(1, n_local_networks + 1):
            # generate a local network
            l_var_intern = list(range(v_cont_var, v_cont_var + n_var_network))
            o_local_network = LocalNetwork(v_num_network, l_var_intern)
            l_local_networks.append(o_local_network)
            v_cont_var = v_cont_var + n_var_network

        # GENERATE COUPLING SIGNALS IN ONE AUXILIARY LIST
        # how many coupling signals will they have RANDOM
        # n_signals_local_network = randint(1, n_relations)
        # Fixed number of coupling signals, fixed in 2
        # we create a list to choose the neighboring networks
        # select the neighboring network
        # generate the list of coupling variables
        # FUTURE JOB!!!
        # generate the coupling function
        # coupling_function = " & ".join( list(map(str, l_output_variables)))
        # coupling_function = "|".join( list(map(str, l_output_variables)))
        # We validate if we have one or several output variables

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
                l_coupling_signals.append(o_coupling_signal)
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
            description_variables = []
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
                description_variables.append(o_variable_model)
                # adding the description in functions of every variable
            # adding the local network to list of local networks
            o_local_network.description_variables = description_variables.copy()
            aux2_l_local_networks.append(o_local_network)
            # actualized the list of local networks
        l_local_networks = aux2_l_local_networks.copy()

        for o_local_network in l_local_networks:
            o_local_network.process_parameters()
            o_local_network.show()
            print("Local network created")
        o_cbn = CBN(l_local_networks, l_coupling_signals)
        return o_cbn

    def find_attractor_fields(self):
        # create a directed graph
        o_graph = nx.DiGraph()

        # add edges to the graph
        for o_local_network in self.l_local_networks:
            for o_input_signal in o_local_network.l_input_signals:
                print("Edge:", o_input_signal.local_network_output,"-", o_input_signal.local_network_input)
                o_graph.add_edge(o_input_signal.local_network_output, o_input_signal.local_network_input)

        # graph have cycles or not
        is_acyclic = nx.is_directed_acyclic_graph(o_graph)
        if is_acyclic:
            # make topological order
            topological_order = list(nx.topological_sort(o_graph))
            print("topological order:", topological_order)
        else:
            print("The graph is cycled you have to use other strategy")

    def show_attractors_fields(self):
        pass

    #
    # # find the number of input Coupling Signals
    # for o_local_network in self.l_local_networks:
    #     print(len(o_local_network.l_input_signals))

