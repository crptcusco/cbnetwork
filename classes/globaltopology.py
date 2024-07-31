# Internal imports
import time
import random

# External imports
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mco
from classes.utils.customtext import CustomText


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
        """
        Initializes the global topology with the specified type and edges.
        """
        self.v_topology = v_topology
        self.l_edges = l_edges
        self.o_graph = nx.DiGraph()
        self.o_graph.add_edges_from(self.l_edges)
        self.d_network_color = {}
        self.generate_local_nets_colors()

    @classmethod
    def show_allowed_topologies(cls):
        """
        Displays the allowed topologies for Directed Graphs.
        """
        CustomText.make_sub_title("List of allowed topologies of Directed Graphs")
        for key, value in cls.allowed_topologies.items():
            print(f"{key} - {value}")

    @classmethod
    def generate_sample_topology(cls, v_topology, n_nodes, n_edges=None):
        """
        Generates a global topology based on the specified type.
        :param v_topology: Type of topology to generate.
        :param n_nodes: Number of nodes in the topology.
        :param n_edges: Number of edges in the topology (if applicable).
        :return: Instance of the specific topology class.
        """
        if v_topology not in cls.allowed_topologies:
            print('ERROR: Not permitted option')
            return None
        if n_nodes <= 1:
            print('ERROR: Number of nodes must be greater than 1')
            return None

        if v_topology == 1:
            return CompleteDigraph(n_nodes=n_nodes)
        elif v_topology == 2:
            return AleatoryFixedDigraph(n_nodes=n_nodes, n_edges=n_edges)
        elif v_topology == 3:
            return CycleDigraph(n_nodes=n_nodes)
        elif v_topology == 4:
            return PathDigraph(n_nodes=n_nodes)
        # Placeholders for additional topologies
        elif v_topology == 5:
            pass
        elif v_topology == 6:
            pass

        return None

    def generate_local_nets_colors(self):
        """
        Generates random colors for local networks.
        """
        l_colors = list(mco.CSS4_COLORS.keys())
        random.shuffle(l_colors)
        for i, color in enumerate(l_colors):
            self.d_network_color[i] = color

    def plot_topology(self, ax=None):
        """
        Plots the topology using matplotlib.
        :param ax: Matplotlib axis to plot on; if None, use the current axis.
        """
        if ax is None:
            ax = plt.gca()

        pos = nx.random_layout(self.o_graph) if self.v_topology == 1 else nx.circular_layout(self.o_graph)
        node_colors = [self.d_network_color.get(node, 'skyblue') for node in self.o_graph.nodes()]
        nx.draw_networkx_nodes(self.o_graph, pos, node_color=node_colors, node_size=500, ax=ax)
        nx.draw_networkx_labels(self.o_graph, pos, font_size=12, font_color='black', ax=ax)
        nx.draw_networkx_edges(self.o_graph, pos, arrows=True, ax=ax)

        ax.set_title("CBN Topology")
        ax.axis("off")

    def get_edges(self):
        """
        Returns the list of edges in the graph.
        """
        return self.l_edges

    def get_nodes(self):
        """
        Returns the set of nodes in the graph.
        """
        return set(self.o_graph.nodes())


class PathDigraph(GlobalTopology):
    def __init__(self, n_nodes):
        """
        Initializes a path graph with the given number of nodes.
        """
        G = nx.path_graph(n_nodes, create_using=nx.DiGraph())
        G = nx.relabel_nodes(G, {i: i + 1 for i in range(n_nodes)})
        l_edges = list(G.edges())
        self.n_nodes = n_nodes
        super().__init__(v_topology=4, l_edges=l_edges)

    def add_node(self):
        """
        Adds a new node to the path graph and maintains the linear structure.
        """
        new_node = self.n_nodes + 1
        last_node = self.n_nodes
        self.o_graph.add_node(new_node)
        self.o_graph.add_edge(last_node, new_node)
        self.l_edges = list(self.o_graph.edges())
        self.n_nodes += 1


class CycleDigraph(GlobalTopology):
    def __init__(self, n_nodes):
        """
        Initializes a cycle graph with the given number of nodes.
        """
        G = nx.cycle_graph(n_nodes, create_using=nx.DiGraph())
        G = nx.relabel_nodes(G, {i: i + 1 for i in range(n_nodes)})
        l_edges = list(G.edges())
        self.n_nodes = n_nodes
        super().__init__(v_topology=3, l_edges=l_edges)

    def add_node(self):
        """
        Adds a new node to the cycle graph and maintains the cyclic structure.
        """
        new_node = self.n_nodes + 1
        last_node = self.n_nodes
        first_node = 1
        self.o_graph.remove_edge(last_node, first_node)
        self.o_graph.add_node(new_node)
        self.o_graph.add_edge(last_node, new_node)
        self.o_graph.add_edge(new_node, first_node)
        self.l_edges = list(self.o_graph.edges())
        self.n_nodes += 1


class CompleteDigraph(GlobalTopology):
    def __init__(self, n_nodes):
        """
        Initializes a complete graph with the given number of nodes.
        """
        G = nx.complete_graph(n_nodes, create_using=nx.DiGraph())
        G = nx.relabel_nodes(G, {i: i + 1 for i in G.nodes()})
        l_edges = list(G.edges())
        self.n_nodes = n_nodes
        super().__init__(v_topology=1, l_edges=l_edges)

    def add_node(self):
        """
        Adds a new node to the complete graph and connects it to all existing nodes.
        """
        new_node = max(self.o_graph.nodes()) + 1
        self.o_graph.add_node(new_node)
        for node in self.o_graph.nodes():
            if node != new_node:
                self.o_graph.add_edge(new_node, node)
                self.o_graph.add_edge(node, new_node)
        self.l_edges = list(self.o_graph.edges())
        self.n_nodes += 1


class AleatoryFixedDigraph(GlobalTopology):
    def __init__(self, n_nodes, n_edges=None):
        """
        Initializes a random directed graph with fixed edges.
        """
        self.n_nodes = n_nodes
        self.l_nodes = list(range(1, n_nodes + 1))
        self.n_edges = n_edges if n_edges is not None else n_nodes
        self.l_edges = []
        self.generate_edges()
        super().__init__(v_topology=2, l_edges=self.l_edges)

    def generate_edges(self):
        """
        Generates edges for the random directed graph.
        """
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

    def add_edge(self):
        """
        Adds a new edge to the graph while maintaining constraints.
        """
        G = nx.DiGraph()
        G.add_nodes_from(self.l_nodes)
        G.add_edges_from(self.l_edges)

        while True:
            u, v = random.sample(self.l_nodes, 2)
            if G.in_degree(v) < 2 and not G.has_edge(u, v):
                G.add_edge(u, v)
                self.l_edges.append((u, v))
                break

        self.n_edges = len(self.l_edges)
        self.update_parent_graph()

    def add_node(self):
        """
        Adds a new node to the graph and updates the edges.
        """
        start_time = time.time()

        G = nx.DiGraph()
        G.add_nodes_from(self.l_nodes)
        G.add_edges_from(self.l_edges)

        new_node = max(self.l_nodes) + 1
        self.l_nodes.append(new_node)
        G.add_node(new_node)
        print(f"Adding new node: {new_node}")

        edge_to_remove = random.choice(list(G.edges))
        G.remove_edge(*edge_to_remove)
        self.l_edges.remove(edge_to_remove)
        print(f"Removed edge: {edge_to_remove}")

        while True:
            u = random.choice(self.l_nodes[:-1])
            if not G.has_edge(u, new_node):
                G.add_edge(u, new_node)
                self.l_edges.append((u, new_node))
                print(f"Added edge from {u} to {new_node}")
                break

        while True:
            v = random.choice(self.l_nodes[:-1])
            if G.in_degree(v) < 2 and not G.has_edge(new_node, v):
                G.add_edge(new_node, v)
                self.l_edges.append((new_node, v))
                print(f"Added edge from {new_node} to {v}")
                break

        for node in self.l_nodes:
            if G.in_degree(node) == 0 and G.out_degree(node) == 0:
                u = random.choice([n for n in self.l_nodes if n != node])
                G.add_edge(u, node)
                self.l_edges.append((u, node))
                print(f"Ensured connectivity by adding edge from {u} to {node}")

        self.n_nodes += 1
        self.n_edges = len(self.l_edges)
        self.update_parent_graph()

        end_time = time.time()
        print(f"Node {new_node} added in {end_time - start_time} seconds")

    def update_parent_graph(self):
        """
        Updates the graph representation after modifications.
        """
        self.o_graph = nx.DiGraph()
        self.o_graph.add_nodes_from(self.l_nodes)
        self.o_graph.add_edges_from(self.l_edges)

    def get_edges(self):
        """
        Returns the list of edges in the graph.
        """
        return self.l_edges
