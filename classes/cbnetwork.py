from random import randint  # generate random numbers integers


class CBN:
    def __init__(self, l_local_networks, l_coupling_signals):
        self.l_local_networks = l_local_networks
        self.l_coupling_signals = l_coupling_signals

    def show(self):
        pass

    @staticmethod
    def generate_aleatory_cbn(n_local_networks, n_var_network, n_relations, relations_fixed=False):
        # generate the local networks in basic form (without relations and dynamic)
        l_local_networks = []
        v_cont_var = 1
        for v_num_network in range(1, n_local_networks + 1):
            # generate a local network
            l_var_intern = list(range(v_cont_var, v_cont_var + n_var_network))
            o_local_network = LocalNetwork(v_num_network, l_var_intern)
            l_local_networks.append(o_local_network)
            v_cont_var = v_cont_var + n_var_network
            l_local_networks.append(o_local_network)
            # o_local_network.show()

        # generate the input relations
        l_coupling_signals = []
        for o_local_network in l_local_networks:
            # Define the number of input relations
            v_aux_n_relations = 1
            if relations_fixed:
                v_aux_n_relations = n_relations
            else:
                v_aux_n_relations = randint(1, n_relations)
            for v_relation in range(1, v_aux_n_relations):
                o_coupling_signal = CouplingSignal()
