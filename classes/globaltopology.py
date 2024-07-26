# internal imports
from classes.utils.customtext import CustomText

# external imports
import random
import networkx as nx  # generate networks
import matplotlib.pyplot as plt  # library to make draws
import matplotlib.colors as mco  # library who have the list of colors


class AleatoryFixedDigraph:
    def __init__(self, n_nodes, n_edges=None, o_base_global_topology=None):
        self.l_edges = []

        if o_base_global_topology is None:
            self.n_nodes = n_nodes
            self.l_nodes = list(range(1, n_nodes + 1))
            self.n_edges = n_edges if n_edges is not None else n_nodes
            self.generate_edges()
        else:
            self.l_nodes = list(range(1, max(o_base_global_topology.get_nodes()) + 1))
            self.l_edges = o_base_global_topology.get_edges()
            self.n_nodes = len(self.l_nodes)
            self.n_edges = len(self.l_edges) + 1  # Adding one new edge

            print(o_base_global_topology.get_edges())
            # Add one new edge
            self.add_edge(o_base_global_topology)

    def generate_edges(self):
        G = nx.DiGraph()
        G.add_nodes_from(self.l_nodes)

        for i in range(1, self.n_nodes):
            u = random.randint(0, i - 1)
            G.add_edge(u, i)

        while G.number_of_edges() < self.n_edges:
            u, v = random.sample(range(self.n_nodes), 2)
            if G.in_degree(v) < 2 and not G.has_edge(u, v):
                G.add_edge(u, v)

        mapping = {node: node + 1 for node in G.nodes()}
        G = nx.relabel_nodes(G, mapping)

        self.l_edges = list(G.edges())

    def get_edges(self):
        return self.l_edges

    def add_edge(self, base_graph):
        self.l_edges = base_graph.get_edges()
        self.l_nodes = list(base_graph.get_nodes())

        G = nx.DiGraph()
        G.add_nodes_from(self.l_nodes)
        G.add_edges_from(self.l_edges)

        while True:
            u, v = random.sample(self.l_nodes, 2)
            if G.in_degree(v) < 2 and not G.has_edge(u, v):
                G.add_edge(u, v)
                self.l_edges.append((u, v))
                break

    def add_edges_for_new_nodes(self, n_nodes, o_base_global_topology):
        """
        Agregar aristas para los nuevos nodos cuando se aumenta el número de nodos en el grafo.
        """
        existing_nodes = len(o_base_global_topology.get_nodes())
        new_nodes = list(range(existing_nodes + 1, n_nodes + 1))
        self.l_nodes.extend(new_nodes)

        G = nx.DiGraph()
        G.add_nodes_from(self.l_nodes)
        G.add_edges_from(self.l_edges)

        # Asegurar que el grafo esté conectado con los nuevos nodos
        for new_node in new_nodes:
            u = random.choice(self.l_nodes[:existing_nodes])
            G.add_edge(u, new_node)
            self.l_edges.append((u, new_node))

        while G.number_of_edges() < self.n_edges:
            u, v = random.sample(self.l_nodes, 2)
            if G.in_degree(v) < 2 and not G.has_edge(u, v):
                G.add_edge(u, v)
                self.l_edges.append((u, v))


class CompleteDigraph:
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        # Create a complete graph with n_nodes
        G = nx.complete_graph(self.n_nodes, nx.DiGraph())
        # Adjust the indices to start from 1
        G = nx.relabel_nodes(G, {i: i + 1 for i in range(9)})
        self.l_edges = list(G.edges())

    def get_edges(self):
        return self.l_edges


class GlobalTopology:
    allowed_topologies = {
        1: "complete",
        2: "aleatory_fixed_2_input_edges",
        3: "cycle",
        4: "path",
        5: "aleatory_gn",
        6: "aleatory_gnc"
    }

    def __init__(self, v_topology, l_edges):
        self.v_topology = v_topology
        self.l_edges = l_edges

        # Create the networkx graph
        self.o_graph = nx.DiGraph()
        self.o_graph.add_edges_from(self.l_edges)

        # Dictionary with the colors
        self.d_network_color = {}
        # Generate the colors for every local network
        self.generate_local_nets_colors()

    @classmethod
    def show_allowed_topologies(cls):
        """
        Display the allowed topologies.
        """
        CustomText.make_sub_title("List of allowed topologies of Directed Graphs")
        for key, value in cls.allowed_topologies.items():
            print(key, "-", value)

    @classmethod
    def generate_sample_topology(cls, v_topology, n_nodes, n_edges=None, o_base_global_topology=None):
        l_edges = []

        if v_topology not in cls.allowed_topologies.keys():
            print('ERROR: Not permitted option')
            return l_edges
        if n_nodes <= 1:
            print('ERROR: Number of nodes less or equal to 1')
            return l_edges

        # Generate edges based on the selected topology
        if v_topology == 1:
            o_complete_digraph = CompleteDigraph(n_nodes=n_nodes)
            l_edges = o_complete_digraph.get_edges()

        elif v_topology == 2:
            o_aleatory_fixed = AleatoryFixedDigraph(n_nodes=n_nodes, n_edges=n_edges,
                                                    o_base_global_topology=o_base_global_topology)
            l_edges = o_aleatory_fixed.get_edges()

        elif v_topology == 3:
            G = nx.cycle_graph(n_nodes, nx.DiGraph())
            G = nx.relabel_nodes(G, {i: i + 1 for i in range(n_nodes)})
            l_edges = list(G.edges())

        elif v_topology == 4:
            G = nx.path_graph(n_nodes, nx.DiGraph())
            G = nx.relabel_nodes(G, {i: i + 1 for i in range(n_nodes)})
            l_edges = list(G.edges())

        elif v_topology == 5:
            # NetworkX does not have a direct aleatory_gn method
            # You need to define this or use another method
            pass

        elif v_topology == 6:
            # NetworkX does not have a direct aleatory_gnc method
            # You need to define this or use another method
            pass

        o_global_topology = GlobalTopology(v_topology=v_topology, l_edges=l_edges)
        return o_global_topology

    def generate_local_nets_colors(self):
        l_colors = list(mco.CSS4_COLORS.keys())
        random.shuffle(l_colors)
        for i, color in enumerate(l_colors):
            self.d_network_color[i] = color

    def plot_topology(self, ax=None):
        if ax is None:
            ax = plt.gca()

        if self.v_topology == 1:
            pos = nx.random_layout(self.o_graph)
        else:
            pos = nx.circular_layout(self.o_graph)

        node_colors = [self.d_network_color.get(node, 'skyblue') for node in self.o_graph.nodes()]
        nx.draw_networkx_nodes(self.o_graph, pos, node_color=node_colors, node_size=500, ax=ax)
        nx.draw_networkx_labels(self.o_graph, pos, font_size=12, font_color='black', ax=ax)
        nx.draw_networkx_edges(self.o_graph, pos, arrows=True, ax=ax)

        ax.set_title("CBN Topology")
        ax.axis("off")

    def get_edges(self):
        return list(self.o_graph.edges())

    def get_nodes(self):
        return set(self.o_graph.nodes())
