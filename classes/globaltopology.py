import random
import networkx as nx  # generate networks
import matplotlib.pyplot as plt  # library to make draws
import matplotlib.colors as mco  # library who have the list of colors

from classes.utils.customtext import CustomText


class GlobalTopology:
    allowed_topologies = {
        1: "complete_graph",
        2: "binomial_tree",
        3: "cycle_graph",
        4: "path_graph",
        5: "gn_graph",
        6: "gnc_graph",
        7: "linear_graph",
        8: "aleatory_digraph"
    }

    def __init__(self, n_nodes, v_topology=1, n_max_edges=None, o_graph=None, seed=None):
        """
        Initialize the CBNTopology class.

        :param seed: Optional; Seed for random number generator.
        """

        self.n_nodes = n_nodes
        self.v_topology = v_topology
        self.n_max_edges = n_max_edges
        self.o_graph = o_graph  # A networkx Graph object to make the visualizations

        self.d_network_color = {}  # Dictionary with the colors
        self.generate_local_nets_colors()  # Generate the colors for every local network

        self.seed = seed
        if seed is not None:
            random.seed(seed)

    @classmethod
    def show_allowed_topologies(cls):
        """
        Display the allowed topologies.
        """
        CustomText.make_sub_title("List of allowed topologies")
        for key, value in cls.allowed_topologies.items():
            print(key, "-", value)

    def generate_local_nets_colors(self):
        # generate a list of colors for the local networks
        l_colors = list(mco.CSS4_COLORS.keys())
        random.shuffle(l_colors)
        for i, color in enumerate(l_colors):
            self.d_network_color[i] = color

    def generate_networkx_graph(self):
        """
        Generate the global topology based on the selected topology type.
        :return: List of edges in the generated graph.
        """
        topology_generators = {
            1: nx.complete_graph,
            2: nx.binomial_tree,
            3: nx.cycle_graph,
            4: nx.path_graph,
            5: nx.gn_graph,
            6: nx.gnc_graph,
            7: GlobalTopology.generate_linear_digraph,
            8: GlobalTopology.generate_aleatory_digraph
        }

        if self.v_topology not in topology_generators:
            raise ValueError(f"Invalid topology type: {self.v_topology}")

        if self.v_topology in [1, 2, 3, 4]:
            o_graph = topology_generators[self.v_topology](self.n_nodes, nx.DiGraph())
        elif self.v_topology in [5, 6]:
            o_graph = topology_generators[self.v_topology](self.n_nodes)
        elif self.v_topology == 7:
            o_graph = self.generate_linear_digraph()
        else:
            o_graph = self.generate_aleatory_digraph()

        # Renaming the label of the nodes for beginning in 1
        mapping = {node: node + 1 for node in o_graph.nodes()}
        o_graph = nx.relabel_nodes(o_graph, mapping)

        self.generate_local_nets_colors()
        self.o_graph = o_graph

    def generate_linear_digraph(self):
        """
        Generate a linear directed graph.
        :return: Directed graph.
        """
        o_graph = nx.DiGraph()
        o_graph.add_nodes_from(range(1, self.n_nodes + 1))
        for i in range(1, self.n_nodes):
            o_graph.add_edge(i, i + 1)
        return o_graph

    def generate_aleatory_digraph(self):
        """
        Generate a random directed graph with a maximum of two incoming edges per node.
        :return: Directed graph.
        """

        if self.n_max_edges is None:
            self.n_max_edges = random.randint(self.n_nodes - 1, 2 * self.n_nodes)
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n_nodes))

        # Ensure the graph is connected by creating a spanning tree
        for i in range(1, self.n_nodes):
            u = random.randint(0, i - 1)
            G.add_edge(u, i)

        # Add additional edges randomly while ensuring no more than two incoming edges per node
        while G.number_of_edges() < self.n_max_edges:
            u = random.randint(0, self.n_nodes - 1)
            v = random.randint(0, self.n_nodes - 1)

            if u != v and G.in_degree(v) < 2 and not G.has_edge(u, v):
                G.add_edge(u, v)

        return G

    def plot_topology(self):
        # Posiciones de los nodos para un gráfico visual más limpio
        pos = nx.circular_layout(self.o_graph)

        # Dibujar nodos y aristas
        # Se agregan colores a los nodos utilizando el diccionario d_network_color
        node_colors = [self.d_network_color.get(node, 'skyblue') for node in self.o_graph.nodes()]
        nx.draw_networkx_nodes(self.o_graph, pos, node_color=node_colors)
        nx.draw_networkx_labels(self.o_graph, pos)
        nx.draw_networkx_edges(self.o_graph, pos, arrows=True)

        plt.title("CBN Topology")
        plt.axis("off")
        plt.show()

    def get_edges(self):
        return list(self.o_graph.edges())
