class CBN(object):
    def __init__(self, l_vertices, l_edges):
        self.l_vertices = l_vertices
        self.l_edges = l_edges

    @staticmethod
    def generate_aleatory_cbn(n_local_networks, n_relations, relations_fixed=False):
        # n_local_networks = n_local_networks
        # n_relations = n_relations
        # relations_fixed = relations_fixed

        # generate the local networks
        for v_num in range(1, n_local_networks+1):
            print(v_num)
            # generate a local network

