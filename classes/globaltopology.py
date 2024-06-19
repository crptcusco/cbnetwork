import random
import networkx as nx  # generate networks
import matplotlib.pyplot as plt  # library to make draws
import matplotlib.colors as mco  # library who have the list of colors

from classes.utils.customtext import CustomText


class GlobalTopology:
    allowed_topologies = {
        1: "complete_graph",
        2: "aleatory_digraph",
        3: "cycle_graph",
        4: "path_graph",
        5: "gn_graph",
        6: "gnc_graph"
    }

    def __init__(self, n_nodes, v_topology=1, n_edges=None, o_graph=None, seed=None):
        """
        Initialize the CBNTopology class.

        :param seed: Optional; Seed for random number generator.
        """

        self.n_nodes = n_nodes
        self.v_topology = v_topology
        self.n_edges = n_edges  # the number of edges that have the graph
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

        if self.o_graph is not None:
            print("entrouuuu")
            self.add_aleatory_edge()

        else:
            topology_generators = {
                1: nx.complete_graph,
                2: self.generate_aleatory_digraph,
                3: nx.cycle_graph,
                4: nx.path_graph,
                5: nx.gn_graph,
                6: nx.gnc_graph
            }

            if self.v_topology not in topology_generators:
                raise ValueError(f"Invalid topology type: {self.v_topology}")
            if self.v_topology in [3, 4]:
                o_graph = topology_generators[self.v_topology](self.n_nodes, nx.DiGraph())
            elif self.v_topology == 2:
                o_graph = topology_generators[self.v_topology]()
            elif self.v_topology in [5, 6]:
                o_graph = topology_generators[self.v_topology](self.n_nodes)
            else:
                o_graph = topology_generators[1](self.n_nodes, nx.DiGraph())

            # Renaming the label of the nodes for beginning in 1
            mapping = {node: node + 1 for node in o_graph.nodes()}
            o_graph = nx.relabel_nodes(o_graph, mapping)

            self.generate_local_nets_colors()
            self.o_graph = o_graph

    def generate_aleatory_digraph(self):
        """
        Generate a random directed graph with a maximum of two incoming edges per node.
        :return: Directed graph.
        """
        n_nodes = self.n_nodes  # Assuming this should be used instead of self.n_nodes

        # Validate if the number of edges is None
        if self.n_edges is None:
            self.n_edges = random.randint(n_nodes - 1, 2 * n_nodes)

        # Validate if the number of edges is more than double the number of nodes
        if self.n_edges > n_nodes * 2:
            self.n_edges = n_nodes * 2
            CustomText.send_warning('Changing the number of edges by excess')

        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))

        # Ensure the graph is connected by creating a spanning tree
        for i in range(1, n_nodes):
            u = random.randint(0, i - 1)
            G.add_edge(u, i)

        # Add additional edges randomly while ensuring no more than two incoming edges per node
        while G.number_of_edges() < self.n_edges:
            u = random.randint(0, n_nodes - 1)
            v = random.randint(0, n_nodes - 1)

            if u != v and G.in_degree(v) < 2 and not G.has_edge(u, v):
                G.add_edge(u, v)

        return G

    def add_aleatory_edge(self):
        """
        Add additional edges randomly while ensuring no more than two incoming edges per node.
        """
        # Print the current edges in the graph
        print("Current edges:", list(self.o_graph.edges))

        # Add additional edges randomly while ensuring no more than two incoming edges per node
        while self.o_graph.number_of_edges() < self.n_edges:
            u = random.randint(1, self.n_nodes)
            v = random.randint(1, self.n_nodes)

            if u != v and self.o_graph.in_degree(v) < 2 and not self.o_graph.has_edge(u, v):
                self.o_graph.add_edge(u, v)

        # Print the updated edges in the graph
        print("Updated edges:", list(self.o_graph.edges))

    def plot_topology(self):
        # Positions of nodes for a cleaner visual graph
        if self.v_topology == 1:
            pos = nx.random_layout(self.o_graph)
        else:
            pos = nx.circular_layout(self.o_graph)

        # Draw nodes and edges
        # Add colors to the nodes using the dictionary d_network_color
        node_colors = [self.d_network_color.get(node, 'skyblue') for node in self.o_graph.nodes()]
        nx.draw_networkx_nodes(self.o_graph, pos, node_color=node_colors, node_size=500)
        nx.draw_networkx_labels(self.o_graph, pos, font_size=12, font_color='black')
        nx.draw_networkx_edges(self.o_graph, pos, arrows=True)

        # Optional: draw edge labels
        # edge_labels = nx.get_edge_attributes(self.o_graph, 'weight')
        # nx.draw_networkx_edge_labels(self.o_graph, pos, edge_labels=edge_labels)

        plt.title("CBN Topology")
        plt.axis("off")
        plt.show()

    def get_edges(self):
        return list(self.o_graph.edges())
