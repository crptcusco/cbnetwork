from random import randint  # generate random numbers integers

from classes.internalvariable import InternalVariable
from classes.localnetwork import LocalNetwork
from classes.couplingsignal import CouplingSignal
import random  # generate random numbers


class CBN:
    def __init__(self, l_local_networks, l_coupling_signals):
        self.l_local_networks = l_local_networks
        self.l_coupling_signals = l_coupling_signals

    def show(self):
        pass

    @staticmethod
    def generate_aleatory_cbn(n_local_networks, n_var_network, n_relations, n_output_variables, n_clauses_function, relations_fixed=False):
        # GENERATE THE LOCAL NETWORKS IN BASIC FORM (WITHOUT RELATIONS AND DYNAMIC)
        l_local_networks = []
        v_cont_var = 1
        for v_num_network in range(1, n_local_networks + 1):
            # generate a local network
            l_var_intern = list(range(v_cont_var, v_cont_var + n_var_network))
            o_local_network = LocalNetwork(v_num_network, l_var_intern)
            l_local_networks.append(o_local_network)
            v_cont_var = v_cont_var + n_var_network
            # o_local_network.show()

        # GENERATE COUPLING SIGNALS IN ONE AUXILIARY LIST
        aux1_l_of_local_networks = []
        for o_local_network in l_local_networks:
            # how many coupling signals will they have RANDOM
            # number_of_signals_local_network = randint(1, n_relations)
            # Fixed number of coupling signals, fixed in 2
            number_of_signals_local_network = n_relations
            # we create a list to choose the neighboring networks
            l_aux_local_networks = l_local_networks.copy()
            l_aux_local_networks.remove(o_local_network)
            # select the neighboring network
            l_local_networks_co = random.sample(l_aux_local_networks, number_of_signals_local_network)
            lista_signals = []
            for o_local_network_co in l_local_networks_co:
                # generate the list of coupling variables
                l_output_variables = random.sample(o_local_network_co.l_var_intern, n_output_variables)

                # FUTURE JOB!!!
                # generate the coupling function
                # coupling_function = " & ".join( list(map(str, l_output_variables)))
                # coupling_function = "|".join( list(map(str, l_output_variables)))

                # We validate if we have one or several output variables
                if n_output_variables == 1:
                    coupling_function = l_output_variables[0]
                else:
                    coupling_function = " " + " ∨ ".join(list(map(str, l_output_variables))) + " "
                o_signal_model = CouplingSignal(o_local_network.index, o_local_network_co.index,
                                                l_output_variables, v_cont_var, coupling_function)
                lista_signals.append(o_signal_model)
                v_cont_var = v_cont_var + 1
            o_local_network.l_input_relations = lista_signals.copy()
            aux1_l_of_local_networks.append(o_local_network)
        l_local_networks = aux1_l_of_local_networks.copy()

        # show the local networks with coupling signals and with description
        # for o_local_network in l_local_networks:
        #    o_local_network.show()

        # GENERATE THE DYNAMICS OF EACH RDD
        number_max_of_clauses = n_clauses_function
        number_max_of_literals = 3
        # we generate an auxiliary list to add the coupling signals
        aux2_l_of_local_networks = []
        for o_local_network in l_local_networks:
            # Create a list of all RDDAs variables
            l_aux_variables = []
            # Add the variables of the coupling signals
            for o_signal in o_local_network.l_input_relations:
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
            aux2_l_of_local_networks.append(o_local_network)
            # actualized the list of local networks
        l_local_networks = aux2_l_of_local_networks.copy()

        for o_local_network in l_local_networks:
            o_local_network.process_parameters()
            o_local_network.show()
            print("Local network created")

        # # generate the input relations
        # l_coupling_signals = []
        # for o_local_network in l_local_networks:
        #     # Define the number of input relations
        #     v_aux_n_relations = 1
        #     if relations_fixed:
        #         v_aux_n_relations = n_relations
        #     else:
        #         v_aux_n_relations = randint(1, n_relations)
        #     l_potential_neighbors = list(map(lambda x :  x.index, l_local_networks))
        #     l_potential_neighbors = l_potential_neighbors.remove(o_local_network.index)
        #
        #     for o_neighbor_network in l_potential_neighbors:
        #         # Calculate the output variables
        #         l_output_variables =
        #
        #         o_coupling_signal = CouplingSignal(local_network_input=o_local_network.index,
        #                                            local_network_output=o_neighbor_network,
        #                                            l_output_variables=,
        #                                            index_variable_signal=,
        #                                            coupling_function=)
