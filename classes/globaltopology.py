# internal imports
from classes.utils.customtext import CustomText

# external imports
import random
import networkx as nx  # generate networks
import matplotlib.pyplot as plt  # library to make draws
import matplotlib.colors as mco  # library who have the list of colors


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
        """
        Genera una topología global basada en el tipo de topología especificado.
        :param v_topology: Tipo de topología a generar.
        :param n_nodes: Número de nodos en la topología.
        :param n_edges: Número de aristas en la topología (si aplica).
        :param o_base_global_topology: Objeto de la topología global base (si aplica).
        :return: Instancia de la clase de topología específica.
        """
        if v_topology not in cls.allowed_topologies.keys():
            print('ERROR: Not permitted option')
            return None
        if n_nodes <= 1:
            print('ERROR: Number of nodes less or equal to 1')
            return None

        if v_topology == 1:
            # Generar un grafo completo
            return CompleteDigraph(n_nodes=n_nodes)

        elif v_topology == 2:
            # Generar un grafo aleatorio con la topología fija
            return AleatoryFixedDigraph(n_nodes=n_nodes, n_edges=n_edges)

        elif v_topology == 3:
            # Generar un grafo cíclico
            return CycleDigraph(n_nodes=n_nodes)  # Devolver una instancia de CycleDigraph

        elif v_topology == 4:
            # Generar un grafo de camino
            return PathDigraph(n_nodes=n_nodes)  # Devolver una instancia de PathDigraph

        elif v_topology == 5:
            # Aquí deberías definir o implementar la topología aleatoria GN
            pass

        elif v_topology == 6:
            # Aquí deberías definir o implementar la topología aleatoria GNC
            pass

        # Si no se encuentra un tipo de topología válido, retornar None
        return None

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
        return self.l_edges

    def get_nodes(self):
        return set(self.o_graph.nodes())


class PathDigraph(GlobalTopology):
    def __init__(self, n_nodes):
        # Crear un grafo de camino
        G = nx.path_graph(n_nodes, nx.DiGraph())
        # Ajustar los índices para que empiecen desde 1
        G = nx.relabel_nodes(G, {i: i + 1 for i in range(n_nodes)})
        l_edges = list(G.edges())
        super().__init__(v_topology=4, l_edges=l_edges)


class CycleDigraph(GlobalTopology):
    def __init__(self, n_nodes):
        # Crear un grafo cíclico
        G = nx.cycle_graph(n_nodes, nx.DiGraph())
        # Ajustar los índices para que empiecen desde 1
        G = nx.relabel_nodes(G, {i: i + 1 for i in range(n_nodes)})
        l_edges = list(G.edges())
        super().__init__(v_topology=3, l_edges=l_edges)


class CompleteDigraph(GlobalTopology):
    def __init__(self, n_nodes):
        # Create a complete graph with n_nodes
        G = nx.complete_graph(n_nodes, create_using=nx.DiGraph())
        # Adjust the indices to start from 1
        G = nx.relabel_nodes(G, {i: i + 1 for i in G.nodes()})
        l_edges = list(G.edges())
        super().__init__(v_topology=1, l_edges=l_edges)  # Assuming v_topology=1 represents complete graph


class AleatoryFixedDigraph(GlobalTopology):
    def __init__(self, n_nodes, n_edges=None):
        self.n_nodes = n_nodes
        self.l_nodes = list(range(1, n_nodes + 1))
        self.n_edges = n_edges if n_edges is not None else n_nodes
        self.l_edges = []
        self.generate_edges()
        super().__init__(v_topology=2, l_edges=self.l_edges)

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

    def add_edge(self):
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
        G = nx.DiGraph()
        G.add_nodes_from(self.l_nodes)
        G.add_edges_from(self.l_edges)

        new_node = max(self.l_nodes) + 1
        self.l_nodes.append(new_node)
        G.add_node(new_node)

        # Remove an edge
        edge_to_remove = random.choice(list(G.edges))
        G.remove_edge(*edge_to_remove)
        self.l_edges.remove(edge_to_remove)

        # Add an edge to the new node
        while True:
            u = random.choice(self.l_nodes[:-1])
            if not G.has_edge(u, new_node):
                G.add_edge(u, new_node)
                self.l_edges.append((u, new_node))
                break

        # Add an edge from the new node
        while True:
            v = random.choice(self.l_nodes[:-1])
            if G.in_degree(v) < 2 and not G.has_edge(new_node, v):
                G.add_edge(new_node, v)
                self.l_edges.append((new_node, v))
                break

        # Ensure all nodes are connected
        for node in self.l_nodes:
            if G.in_degree(node) == 0 and G.out_degree(node) == 0:
                u = random.choice([n for n in self.l_nodes if n != node])
                G.add_edge(u, node)
                self.l_edges.append((u, node))

        self.n_nodes += 1
        self.n_edges = len(self.l_edges)
        self.update_parent_graph()

    def update_parent_graph(self):
        self.o_graph = nx.DiGraph()
        self.o_graph.add_nodes_from(self.l_nodes)
        self.o_graph.add_edges_from(self.l_edges)

    def get_edges(self):
        return self.l_edges
