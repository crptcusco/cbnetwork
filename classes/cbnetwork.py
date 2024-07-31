# external imports
import itertools  # libraries to iterate
from itertools import product  # generate combinations of numbers
from typing import List, Optional, Any, Dict

from classes.globalscene import GlobalScene
from classes.globaltopology import GlobalTopology
from classes.internalvariable import InternalVariable
from classes.localnetwork import LocalNetwork
from classes.directededge import DirectedEdge
from classes.localscene import LocalAttractor
from classes.localtemplates import LocalNetworkTemplate
from classes.utils.customtext import CustomText


class CBN:
    """
    Class representing a Complex Boolean Network (CBN).

    Attributes:
        l_local_networks (list[LocalNetwork]): List of local networks in the CBN.
        l_directed_edges (list[DirectedEdge]): List of directed edges in the CBN.
        d_local_attractors (dict): Dictionary of local attractors.
        d_attractor_pair (dict): Dictionary of attractor pairs.
        d_attractor_fields (dict): Dictionary of attractor fields.
        l_global_scenes (list[GlobalScene]): List of global scenes.
        o_global_topology (GlobalTopology): Global topology object.
    """

    def __init__(self, l_local_networks: list, l_directed_edges: list):
        """
        Initializes the Complex Boolean Network with local networks and directed edges.

        Args:
            l_local_networks (list): List of local networks in the CBN.
            l_directed_edges (list): List of directed edges in the CBN.
        """
        # Initial attributes
        self.l_local_networks = l_local_networks
        self.l_directed_edges = l_directed_edges

        # Calculated attributes
        self.d_local_attractors = {}
        self.d_attractor_pair = {}
        self.d_attractor_fields = {}
        self.l_global_scenes = []

        # Graphs
        self.o_global_topology = None

    def process_output_signals(self) -> None:
        """
        Update output signals for every local network based on directed edges.

        This method iterates over all local networks and directed edges,
        and appends each directed edge to the output signals of the local
        network if the local network's index matches the destination node
        of the directed edge.
        """
        # Create a dictionary for quick lookup of output signals
        local_network_dict = {network.l_index: network for network in self.l_local_networks}

        # Update output signals for each local network
        for edge in self.l_directed_edges:
            source, destination = edge
            if destination in local_network_dict:
                o_local_network = local_network_dict[destination]
                o_local_network.l_output_signals.append(edge)
                print(edge)

    def update_network_by_index(self, o_local_network_update) -> bool:
        """
        Update a local network in the list by its index.

        Args:
            o_local_network_update (LocalNetwork): The local network object with updated information.

        Returns:
            bool: True if the network was found and updated, False otherwise.
        """
        # Iterate over the list of local networks
        for i, o_local_network in enumerate(self.l_local_networks):
            if o_local_network.index == o_local_network_update.index:
                # Update the local network in the list
                self.l_local_networks[i] = o_local_network_update
                print(f"Local Network with index {o_local_network_update.index} updated")
                return True

        # If no network was found, print an error message
        print(f"ERROR: Local Network with index {o_local_network_update.index} not found")
        return False

    def find_local_attractors_sequential(self) -> None:
        """
        Finds local attractors sequentially and updates the list of local attractors in the object.

        This method calculates the local attractors for each local network, updates the coupling signals,
        assigns global indices to each attractor, and generates the attractor dictionary.
        """
        CustomText.make_title('FIND LOCAL ATTRACTORS')

        for o_local_network in self.l_local_networks:
            # Calculate the local attractors for the local network
            l_local_scenes = self._generate_local_scenes(o_local_network)
            o_local_network = LocalNetwork.find_local_attractors(
                o_local_network=o_local_network,
                l_local_scenes=l_local_scenes
            )
            # Update the coupling signals to be analyzed
            self.process_kind_signal(o_local_network)

        # Assign global indices to each attractor
        self._assign_global_indices_to_attractors()

        # Generate the attractor dictionary
        self.generate_attractor_dictionary()

        print('Number of local attractors:', self._count_total_attractors())
        CustomText.make_sub_sub_title('END FIND LOCAL ATTRACTORS')

    def _generate_local_scenes(self, o_local_network: LocalNetwork) -> Optional[List[str]]:
        """
        Generate local scenes for the given local network based on external variables.

        Args:
            o_local_network (LocalNetwork): The local network object.

        Returns:
            Optional[List[str]]: A list of local scenes or None if there are no external variables.
        """
        if len(o_local_network.l_var_exterm) != 0:
            return list(product('01', repeat=len(o_local_network.l_var_exterm)))
        return None

    def _assign_global_indices_to_attractors(self) -> None:
        """
        Assign global indices to each attractor in all local networks.
        """
        i_attractor = 1
        for o_local_network in self.l_local_networks:
            for o_local_scene in o_local_network.l_local_scenes:
                for o_attractor in o_local_scene.l_attractors:
                    o_attractor.g_index = i_attractor
                    i_attractor += 1

    def _count_total_attractors(self) -> int:
        """
        Count the total number of attractors across all local networks.

        Returns:
            int: The total number of attractors.
        """
        return sum(len(o_local_scene.l_attractors) for o_local_network in self.l_local_networks for o_local_scene in
                   o_local_network.l_local_scenes)

    def process_kind_signal(self, o_local_network: LocalNetwork) -> None:
        """
        Update the coupling signals to be analyzed for the given local network.

        Args:
            o_local_network (LocalNetwork): The local network object.
        """

        def get_true_table_index(o_state, o_output_signal):
            true_table_index = ""
            for v_output_variable in o_output_signal.l_output_variables:
                pos = o_local_network.l_var_total.index(v_output_variable)
                value = o_state.l_variable_values[pos]
                true_table_index += str(value)
            return true_table_index

        def update_output_signals(l_signals_in_attractor, o_output_signal, o_attractor):
            output_value = l_signals_in_attractor[0]
            if output_value == '0':
                o_output_signal.d_out_value_to_attractor[0].append(o_attractor)
            elif output_value == '1':
                o_output_signal.d_out_value_to_attractor[1].append(o_attractor)

        l_directed_edges = CBN.find_output_edges_by_network_index(o_local_network.index, self.l_directed_edges)

        for o_output_signal in l_directed_edges:
            l_signals_for_output = []
            for o_local_scene in o_local_network.l_local_scenes:
                l_signals_in_local_scene = []
                for o_attractor in o_local_scene.l_attractors:
                    l_signals_in_attractor = [
                        o_output_signal.true_table[get_true_table_index(o_state, o_output_signal)]
                        for o_state in o_attractor.l_states
                    ]

                    if len(set(l_signals_in_attractor)) == 1:
                        l_signals_in_local_scene.append(l_signals_in_attractor[0])
                        update_output_signals(l_signals_in_attractor, o_output_signal, o_attractor)

                if len(set(l_signals_in_local_scene)) == 1:
                    l_signals_for_output.append(l_signals_in_local_scene[0])
                else:
                    l_signals_for_output.extend(l_signals_in_local_scene)

            signal_set_length = len(set(l_signals_for_output))
            if signal_set_length == 1:
                o_output_signal.kind_signal = 1
                print("INFO: the output signal is restricted")
            elif signal_set_length == 2:
                o_output_signal.kind_signal = 3
                print("INFO: the output signal is stable")
            else:
                o_output_signal.kind_signal = 4
                print("INFO: the scene signal is not stable. This CBN doesn't have stable Attractor Fields")

    def find_compatible_pairs(self) -> None:
        """
        Generate pairs of attractors using the output signal.

        Returns:
            None: Updates the dictionary of compatible attractor pairs in the object.
        """

        CustomText.make_title('FIND COMPATIBLE ATTRACTOR PAIRS')

        def find_attractor_pairs(signal_value, o_output_signal, l_attractors_input):
            """
            Find pairs of attractors based on the input signal value.

            Args:
                signal_value (int): The signal value (0 or 1).
                o_output_signal: The output signal object.
                l_attractors_input (list): List of attractor indices for the input signal.

            Returns:
                list: List of pairs of attractors.
            """
            l_attractors_output = [
                o_attractor.g_index
                for o_attractor in
                self.get_attractors_by_input_signal_value(o_output_signal.index_variable, signal_value)
            ]
            return list(itertools.product(l_attractors_input, l_attractors_output))

        n_pairs = 0

        for o_local_network in self.l_local_networks:
            l_output_edges = self.get_output_edges_by_network_index(o_local_network.index)

            for o_output_signal in l_output_edges:
                l_attractors_input_0 = [attr.g_index for attr in o_output_signal.d_out_value_to_attractor[0]]
                l_attractors_input_1 = [attr.g_index for attr in o_output_signal.d_out_value_to_attractor[1]]

                o_output_signal.d_comp_pairs_attractors_by_value[0] = find_attractor_pairs(0, o_output_signal,
                                                                                           l_attractors_input_0)
                o_output_signal.d_comp_pairs_attractors_by_value[1] = find_attractor_pairs(1, o_output_signal,
                                                                                           l_attractors_input_1)

                n_pairs += len(o_output_signal.d_comp_pairs_attractors_by_value[0])
                n_pairs += len(o_output_signal.d_comp_pairs_attractors_by_value[1])

        print(f'Number of attractor pairs: {n_pairs}')
        CustomText.make_sub_sub_title('END FIND ATTRACTOR PAIRS')

    def order_edges_by_compatibility(self):
        """
        Order the directed edges based on their compatibility.

        The compatibility is determined if the input or output local network of one edge
        matches with the input or output local network of any edge in the base group.
        """

        def is_compatible(l_group_base, o_group):
            """
            Check if the given edge group is compatible with any edge in the base group.

            Args:
                l_group_base (list): List of base edge groups.
                o_group: Edge group to be checked for compatibility.

            Returns:
                bool: True if compatible, False otherwise.
            """
            for aux_par in l_group_base:
                if (aux_par.input_local_network == o_group.input_local_network or
                        aux_par.input_local_network == o_group.output_local_network):
                    return True
                elif (aux_par.output_local_network == o_group.output_local_network or
                      aux_par.output_local_network == o_group.input_local_network):
                    return True
            return False

        # Initialize the base list with the first edge group
        l_base = [self.l_directed_edges[0]]
        aux_l_rest_groups = self.l_directed_edges[1:]

        # Process each remaining edge group
        for v_group in aux_l_rest_groups:
            if is_compatible(l_base, v_group):
                l_base.append(v_group)
            else:
                # If not compatible, move it to the end of the list
                aux_l_rest_groups.remove(v_group)
                aux_l_rest_groups.append(v_group)

        # Combine the base list with the rest of the groups
        self.l_directed_edges = [self.l_directed_edges[0]] + aux_l_rest_groups
        print("Directed Edges ordered.")

    def mount_stable_attractor_fields(self) -> None:
        """
        Assembles compatible attractor fields.

        This function assembles fields of attractors that are compatible with each other.
        """

        def evaluate_pair(base_pairs: list, candidate_pair: tuple) -> bool:
            """
            Checks if a candidate attractor pair is compatible with a base attractor pair.

            Args:
                base_pairs (list): List of base attractor pairs.
                candidate_pair (tuple): Candidate attractor pair.

            Returns:
                bool: True if the candidate pair is compatible with the base pairs, False otherwise.
            """

            # Extract the indices of local networks from each attractor pair
            base_attractor_indices = {attractor for pair in base_pairs for attractor in pair}

            # Generate the list of already visited networks
            already_visited_networks = {self.d_local_attractors[idx][0] for idx in base_attractor_indices}

            double_check = 0
            for candidate_idx in candidate_pair:
                if self.d_local_attractors[candidate_idx][0] in already_visited_networks:
                    if candidate_idx in base_attractor_indices:
                        double_check += 1
                else:
                    double_check += 1

            return double_check == 2

        def cartesian_product_mod(base_pairs: list, candidate_pairs: list) -> list:
            """
            Performs the modified Cartesian product of the attractor pairs lists.

            Args:
                base_pairs (list): List of base attractor pairs.
                candidate_pairs (list): List of candidate attractor pairs.

            Returns:
                list: List of candidate attractor fields.
            """
            field_pair_list = []

            for base_pair in base_pairs:
                for candidate_pair in candidate_pairs:
                    if isinstance(base_pair, tuple):
                        base_pair = [base_pair]
                    if evaluate_pair(base_pair, candidate_pair):
                        new_pair = base_pair + [candidate_pair]
                        field_pair_list.append(new_pair)

            return field_pair_list

        CustomText.make_title('FIND ATTRACTOR FIELDS')

        # Order the edges by compatibility
        self.order_edges_by_compatibility()

        # Generate the base list of pairs made with 0 or 1 coupling signal
        l_base_pairs = set(self.l_directed_edges[0].d_comp_pairs_attractors_by_value[0] +
                           self.l_directed_edges[0].d_comp_pairs_attractors_by_value[1])

        # Iterate over each edge to form unions with the base
        for o_directed_edge in self.l_directed_edges[1:]:
            l_candidate_pairs = o_directed_edge.d_comp_pairs_attractors_by_value[0] + \
                                o_directed_edge.d_comp_pairs_attractors_by_value[1]
            l_base_pairs = cartesian_product_mod(l_base_pairs, l_candidate_pairs)

            if not l_base_pairs:
                break

        # Generate a dictionary of attractor fields
        self.d_attractor_fields = {}
        for i, base_element in enumerate(l_base_pairs, start=1):
            self.d_attractor_fields[i] = list({item for pair in base_element for item in pair})

        print("Number of attractor fields found:", len(l_base_pairs))
        CustomText.make_sub_sub_title("END MOUNT ATTRACTOR FIELDS")

    # SHOW FUNCTIONS
    def show_directed_edges(self) -> None:
        CustomText.print_duplex_line()
        print("SHOW THE DIRECTED EDGES OF THE CBN")
        for o_directed_edge in self.l_directed_edges:
            o_directed_edge.show()

    def show_coupled_signals_kind(self) -> None:
        CustomText.print_duplex_line()
        print("SHOW THE COUPLED SIGNALS KINDS")
        n_restricted_signals = 0
        for o_directed_edge in self.l_directed_edges:
            print(f"SIGNAL: {o_directed_edge.index_variable}, "
                  f"RELATION: {o_directed_edge.output_local_network} -> {o_directed_edge.input_local_network}, "
                  f"KIND: {o_directed_edge.kind_signal} - {o_directed_edge.d_kind_signal[o_directed_edge.kind_signal]}")
            if o_directed_edge.kind_signal == 1:
                n_restricted_signals += 1
        print("Number of restricted signals:", n_restricted_signals)

    def show_description(self) -> None:
        CustomText.make_title('CBN description')
        l_local_networks_indexes = [o_local_network.index for o_local_network in self.l_local_networks]
        CustomText.make_sub_title(f"Local Networks: {l_local_networks_indexes}")
        for o_local_network in self.l_local_networks:
            o_local_network.show()
        CustomText.make_sub_title(f"Directed edges: {l_local_networks_indexes}")
        # for o_directed_edge in self.l_directed_edges:
        #     o_directed_edge.show()

    def show_global_scenes(self) -> None:
        CustomText.make_sub_title('LIST OF GLOBAL SCENES')
        for o_global_scene in self.l_global_scenes:
            o_global_scene.show()

    def show_local_attractors(self) -> None:
        CustomText.make_title('Show local attractors')
        for o_local_network in self.l_local_networks:
            CustomText.make_sub_title(f"Network {o_local_network.index}")
            for o_scene in o_local_network.l_local_scenes:
                CustomText.make_sub_sub_title(
                    f"Network: {o_local_network.index} - Scene: {o_scene.l_values} - N. of Attractors: {len(o_scene.l_attractors)}")
                print(f"Network: {o_local_network.index} - Scene: {o_scene.l_values}")
                print(f"Attractors number: {len(o_scene.l_attractors)}")
                for o_attractor in o_scene.l_attractors:
                    CustomText.print_simple_line()
                    print(f"Global index: {o_attractor.g_index} -> {self.d_local_attractors[o_attractor.g_index]}")
                    for o_state in o_attractor.l_states:
                        print(o_state.l_variable_values)

    def show_attractor_pairs(self) -> None:
        CustomText.print_duplex_line()
        print("LIST OF THE COMPATIBLE ATTRACTOR PAIRS")
        for o_directed_edge in self.l_directed_edges:
            CustomText.print_simple_line()
            print(f"Edge: {o_directed_edge.output_local_network} -> {o_directed_edge.input_local_network}")
            for key in o_directed_edge.d_comp_pairs_attractors_by_value.keys():
                CustomText.print_simple_line()
                print(f"Coupling Variable: {o_directed_edge.index_variable}, Scene: {key}")
                for o_pair in o_directed_edge.d_comp_pairs_attractors_by_value[key]:
                    print(o_pair)

    def show_stable_attractor_fields(self) -> None:
        CustomText.print_duplex_line()
        print("Show the list of attractor fields")
        print("Number Stable Attractor Fields:", len(self.d_attractor_fields))
        for key, o_attractor_field in self.d_attractor_fields.items():
            CustomText.print_simple_line()
            print(key)
            print(o_attractor_field)

    def show_resume(self) -> None:
        CustomText.make_title('CBN Detailed Resume')
        CustomText.make_sub_sub_title('Principal characteristics')
        CustomText.print_simple_line()
        print('Number of local networks:', len(self.l_local_networks))
        print('Number of variables per local network:', self.get_n_local_variables())
        print('Kind Topology:', self.get_kind_topology())
        print('Number of input variables:', self.get_n_input_variables())
        print('Number of output variables:', self.get_n_output_variables())

        CustomText.make_sub_sub_title("Indicators")
        CustomText.print_simple_line()
        print("Number of local attractors:", self.get_n_local_attractors())
        print("Number of attractor pairs:", self.get_n_pair_attractors())
        print("Number of attractor fields:", self.get_n_attractor_fields())
        CustomText.print_simple_line()

    # GET FUNCTIONS
    def get_network_by_index(self, index: int) -> Optional[LocalNetwork]:
        for o_local_network in self.l_local_networks:
            if o_local_network.l_index == index:
                return o_local_network
        return None

    def get_input_edges_by_network_index(self, index: int) -> List[DirectedEdge]:
        return [o_directed_edge for o_directed_edge in self.l_directed_edges if
                o_directed_edge.input_local_network == index]

    def get_output_edges_by_network_index(self, index: int) -> List[DirectedEdge]:
        return [o_directed_edge for o_directed_edge in self.l_directed_edges if
                o_directed_edge.output_local_network == index]

    def get_index_networks(self) -> List[int]:
        return [i_network.l_index for i_network in self.l_local_networks]

    def get_attractors_by_input_signal_value(self, index_variable_signal: int, signal_value: int) -> List[LocalAttractor]:
        l_attractors = []
        for o_local_network in self.l_local_networks:
            for scene in o_local_network.l_local_scenes:
                if scene.l_values is not None and index_variable_signal in scene.l_index_signals:
                    pos = scene.l_index_signals.index(index_variable_signal)
                    if scene.l_values[pos] == str(signal_value):
                        l_attractors.extend(scene.l_attractors)
        return l_attractors

    def get_n_local_attractors(self) -> int:
        return sum(len(o_scene.l_attractors) for o_local_network in self.l_local_networks for o_scene in
                   o_local_network.l_local_scenes)

    def get_n_pair_attractors(self) -> int:
        return sum(len(o_directed_edge.d_comp_pairs_attractors_by_value[0]) + len(
            o_directed_edge.d_comp_pairs_attractors_by_value[1]) for o_directed_edge in self.l_directed_edges)

    def get_n_attractor_fields(self) -> int:
        return len(self.d_attractor_fields)

    def get_n_local_variables(self) -> int:
        return len(self.l_local_networks[0].l_var_intern) if self.l_local_networks else 0

    def generate_attractor_dictionary(self) -> None:
        """
        Generates a Dictionary of local attractors
        :return: a list of triples (a,b,c) where:
         - 'a' is the network index
         - 'b' is the scene index
         - 'c' is the local attractor index
        """
        d_local_attractors = {}
        for o_local_network in self.l_local_networks:
            for o_scene in o_local_network.l_local_scenes:
                for o_attractor in o_scene.l_attractors:
                    t_triple = (o_local_network.index, o_scene.index, o_attractor.l_index)
                    d_local_attractors[o_attractor.g_index] = t_triple

        self.d_local_attractors = d_local_attractors

    # Método para obtener un atractor local por su índice
    def get_local_attractor_by_index(self, i_attractor: int) -> Optional[LocalAttractor]:
        for o_local_network in self.l_local_networks:
            for o_scene in o_local_network.l_local_scenes:
                for o_attractor in o_scene.l_attractors:
                    if o_attractor.g_index == i_attractor:
                        return o_attractor
        print('ERROR: Attractor index not found')
        return None

    # Método para mostrar el diccionario de atractores locales
    def show_local_attractors_dictionary(self) -> None:
        CustomText.make_title('Global Dictionary of local attractors')
        for key, value in self.d_local_attractors.items():
            print(key, '->', value)

    # Método para mostrar campos de atractores estables de manera detallada
    def show_stable_attractor_fields_detailed(self) -> None:
        CustomText.print_duplex_line()
        print("Show the list of attractor fields")
        print("Number Stable Attractor Fields:", len(self.d_attractor_fields))
        for key, value in self.d_attractor_fields.items():
            CustomText.print_simple_line()
            print(key, '->', value)
            for i_attractor in value:
                print(i_attractor, '->', self.d_local_attractors[i_attractor])
                o_attractor = self.get_local_attractor_by_index(i_attractor)
                if o_attractor:
                    o_attractor.show()

    # Método para mostrar campos de atractores
    def show_attractors_fields(self) -> None:
        CustomText.make_sub_title('List of attractor fields')
        for key, value in self.d_attractor_fields.items():
            print(key, "->", value)
        print(f"Number of attractor fields found: {len(self.d_attractor_fields)}")

    # Método para generar escenas globales
    def generate_global_scenes(self) -> None:
        CustomText.make_title('Generated Global Scenes')
        l_edges_indexes = [o_directed_edge.index_variable for o_directed_edge in self.l_directed_edges]
        binary_combinations = list(product([0, 1], repeat=len(l_edges_indexes)))
        self.l_global_scenes = [GlobalScene(l_edges_indexes, list(combination)) for combination in binary_combinations]
        CustomText.make_sub_title('Global Scenes generated')

    # Método para graficar la topología
    def plot_topology(self, ax=None) -> None:
        self.o_global_topology.plot_topology(ax=ax)

    # Método para contar campos estables por escenas globales
    def count_fields_by_global_scenes(self) -> None:
        CustomText.make_sub_title('Counting the stable attractor fields by global scene')
        d_global_scenes_count: Dict[str, int] = {}
        for key, o_attractor_field in self.d_attractor_fields.items():
            d_variable_value = {}
            for i_attractor in o_attractor_field:
                o_attractor = self.get_local_attractor_by_index(i_attractor)
                if o_attractor:
                    for aux_pos, aux_variable in enumerate(o_attractor.relation_index):
                        d_variable_value[aux_variable] = o_attractor.local_scene[aux_pos]
            sorted_dict = {k: d_variable_value[k] for k in sorted(d_variable_value)}
            combination_key = ''.join(str(sorted_dict[k]) for k in sorted_dict)
            if combination_key in d_global_scenes_count:
                d_global_scenes_count[combination_key] += 1
            else:
                d_global_scenes_count[combination_key] = 1
        print(d_global_scenes_count)

    # # OLD GENERATOR

    #
    # @staticmethod
    # def generate_directed_edges(i_last_variable, l_local_networks, l_relations, n_output_variables=2):
    #     """
    #     Generates directed edges based on local networks and their relations.
    #
    #     Args:
    #         i_last_variable (int): Index of the last variable.
    #         l_local_networks (list): List of LocalNetwork objects.
    #         l_relations (list): List of tuples representing relations between local networks.
    #         n_output_variables (int, optional): Number of output variables. Defaults to 2.
    #
    #     Returns:
    #         list: List of DirectedEdge objects.
    #     """
    #     l_directed_edges = []
    #     i_directed_edge = i_last_variable + 1
    #     index_edge = 1
    #
    #     for o_local_network in l_local_networks:
    #         l_local_networks_co = []
    #         for t_relation in l_relations:
    #             if t_relation[1] == o_local_network.index:
    #                 o_local_network_aux = next(filter(lambda x: x.index == t_relation[0], l_local_networks), None)
    #                 l_local_networks_co.append(o_local_network_aux)
    #
    #         for o_local_network_co in l_local_networks_co:
    #             l_output_variables = random.sample(o_local_network_co.l_var_intern, n_output_variables)
    #             if n_output_variables == 1:
    #                 coupling_function = l_output_variables[0]
    #             else:
    #                 coupling_function = " " + " ∨ ".join(list(map(str, l_output_variables))) + " "
    #
    #             o_directed_edge = DirectedEdge(index_edge, i_directed_edge, o_local_network.index,
    #                                            o_local_network_co.index, l_output_variables, coupling_function)
    #             l_directed_edges.append(o_directed_edge)
    #             i_directed_edge += 1
    #             index_edge += 1
    #
    #     return l_directed_edges
    #

    #

    #
    # @staticmethod
    # def generate_local_networks_variables_dynamic(l_local_networks, l_directed_edges):
    #     """
    #     Generates the dynamics for each local network by creating internal variables and their clauses.
    #
    #     Args:
    #         l_local_networks (list): List of LocalNetwork objects.
    #         l_directed_edges (list): List of DirectedEdge objects.
    #
    #     Returns:
    #         list: List of updated LocalNetwork objects with dynamic variables.
    #     """
    #
    #     # the max number of clauses and literals is fixed in 2 and 3 respectively
    #     NUMBER_MAX_OF_CLAUSES = 2
    #     NUMBER_MAX_OF_LITERALS = 3
    #
    #     l_local_networks_updated = []
    #     for o_local_network in l_local_networks:
    #         l_aux_variables = []
    #         l_input_signals = CBN.find_input_edges_by_network_index(o_local_network.index, l_directed_edges)
    #         for o_signal in l_input_signals:
    #             l_aux_variables.append(o_signal.index_variable)
    #         l_aux_variables.extend(o_local_network.l_var_intern)
    #
    #         des_funct_variables = []
    #         for i_local_variable in o_local_network.l_var_intern:
    #             l_clauses_node = []
    #             for _ in range(randint(1, NUMBER_MAX_OF_CLAUSES)):
    #                 v_num_variable = randint(1, NUMBER_MAX_OF_LITERALS)
    #                 l_literals_variables = random.sample(l_aux_variables, v_num_variable)
    #                 l_clauses_node.append(l_literals_variables)
    #
    #             o_internal_variable = InternalVariable(i_local_variable, l_clauses_node)
    #             des_funct_variables.append(o_internal_variable)
    #
    #         o_local_network.des_funct_variables = des_funct_variables.copy()
    #         l_local_networks_updated.append(o_local_network)
    #         CustomText.make_sub_title(f"Local network created : {o_local_network.index}")
    #
    #     return l_local_networks_updated
    #
    # @staticmethod
    # def generate_aleatory_cbn_by_topology(n_local_networks, n_var_network, v_topology, l_edges=None,
    #                                       n_output_variables=2, n_input_variables=2, n_edges=None):
    #     """
    #     Generates an instance of a CBN.
    #
    #     Args:
    #         n_local_networks (int): Number of local networks to generate.
    #         n_var_network (int): Number of variables per network.
    #         v_topology (int): Type of topology for the CBN.
    #         l_edges (list): List of global edges.
    #         n_output_variables (int): Number of output variables.
    #         n_input_variables (int): Number of input variables.
    #         n_edges (int): Number of edges in the CBN.
    #         local_template (AleatoryTemplate): Template for local networks.
    #
    #     Returns:
    #         CBN: The generated CBN object.
    #     """
    #
    #     CustomText.make_title('CBN GENERATION')
    #
    #     # generate the CBN topology
    #     o_global_topology = GlobalTopology.generate_sample_topology(v_topology=v_topology, n_nodes=n_local_networks)
    #
    #     # generate the local network dynamic
    #
    #     # generate the local networks with the indexes and variables (without relations or dynamics)
    #     l_local_networks = CBN.generate_local_networks_indexes_variables(n_local_networks, n_var_network)
    #
    #     l_relations = o_global_topology.get_edges()
    #
    #     # search the last variable from the local network variables
    #     i_last_variable = l_local_networks[-1].l_var_intern[-1]
    #
    #     # generate the directed edges given the last variable generated
    #     l_directed_edges = CBN.generate_directed_edges(i_last_variable=i_last_variable,
    #                                                    l_local_networks=l_local_networks,
    #                                                    l_relations=l_relations,
    #                                                    n_output_variables=n_output_variables)
    #
    #     # Process the coupling signals for every local network
    #     for o_local_network in l_local_networks:
    #         # find the signals for every local network
    #         l_input_signals = CBN.find_input_edges_by_network_index(o_local_network.index, l_directed_edges)
    #         o_local_network.process_input_signals(l_input_signals)
    #
    #     l_local_networks = CBN.generate_local_networks_variables_dynamic(l_local_networks=l_local_networks,
    #                                                                      l_directed_edges=l_directed_edges)
    #
    #     # create the cbn object
    #     o_cbn = CBN(l_local_networks, l_directed_edges)
    #     # add the Global Topology Object
    #     o_cbn.o_global_topology = o_global_topology
    #     return o_cbn

    # NEW GENERATOR
    @staticmethod
    def cbn_generator(
            v_topology: int,
            n_local_networks: int,
            n_vars_network: int,
            n_input_variables: int,
            n_output_variables: int,
            n_max_of_clauses: Optional[int] = None,
            n_max_of_literals: Optional[int] = None,
            n_edges: Optional[int] = None
    ) -> 'CBN':
        """
        Generates a special CBN based on the provided parameters.

        Args:
            v_topology (str): The topology of the CBN. Can be 'aleatory' or other valid types.
            n_local_networks (int): The number of local networks in the CBN.
            n_vars_network (int): The number of variables per local network.
            n_input_variables (int): The number of input variables in the CBN.
            n_output_variables (int): The number of output variables in the CBN.
            n_max_of_clauses (Optional[int]): The maximum number of clauses for the local networks. Defaults to None.
            n_max_of_literals (Optional[int]): The maximum number of literals for the local networks. Defaults to None.
            n_edges (Optional[int]): The number of edges between local networks. Defaults to None.

        Returns:
            CBN: A CBN instance generated based on the given parameters.
        """

        # GENERATE THE GLOBAL TOPOLOGY
        o_global_topology = GlobalTopology.generate_sample_topology(
            v_topology=v_topology,
            n_nodes=n_local_networks,
            n_edges=n_edges
        )

        # GENERATE THE LOCAL NETWORK TEMPLATE
        o_template = LocalNetworkTemplate(
            v_topology=v_topology,
            n_vars_network=n_vars_network,
            n_input_variables=n_input_variables,
            n_output_variables=n_output_variables,
            n_max_of_clauses=n_max_of_clauses,
            n_max_of_literals=n_max_of_literals
        )

        # GENERATE THE CBN WITH THE TOPOLOGY AND TEMPLATE
        o_cbn = CBN.generate_cbn_from_template(
            v_topology=v_topology,
            n_local_networks=n_local_networks,
            n_vars_network=n_vars_network,
            o_template=o_template,
            l_global_edges=o_global_topology.l_edges
        )

        return o_cbn

    @staticmethod
    def find_output_edges_by_network_index(
            index: int,
            l_directed_edges: List['DirectedEdge']
    ) -> List['DirectedEdge']:
        """
        Finds all output edges for a given network index.

        Args:
            index (int): The index of the local network.
            l_directed_edges (List[DirectedEdge]): List of DirectedEdge objects.

        Returns:
            List[DirectedEdge]: List of DirectedEdge objects that are output edges for the specified network index.
        """
        return [edge for edge in l_directed_edges if edge.output_local_network == index]

    @staticmethod
    def find_input_edges_by_network_index(index, l_directed_edges):
        """
        Finds all input edges for a given network index.

        Args:
            index (int): The index of the local network.
            l_directed_edges (list): List of DirectedEdge objects.

        Returns:
            list: List of DirectedEdge objects that are input edges for the specified network index.
        """
        return [edge for edge in l_directed_edges if edge.input_local_network == index]

    @staticmethod
    def generate_local_networks_indexes_variables(n_local_networks, n_vars_network):
        """
        Generates local networks and their variable indexes.

        Args:
            n_local_networks (int): Number of local networks to generate.
            n_vars_network (int): Number of variables per network.

        Returns:
            list: List of LocalNetwork objects.
        """
        l_local_networks = []
        v_cont_var = 1
        for v_num_network in range(1, n_local_networks + 1):
            # generate the variables of the networks
            l_var_intern = list(range(v_cont_var, v_cont_var + n_vars_network))
            # create the Local Network object
            o_local_network = LocalNetwork(v_num_network, l_var_intern)
            # add the local network object to the list
            l_local_networks.append(o_local_network)
            # update the index of the variables
            v_cont_var += n_vars_network
        return l_local_networks

    @staticmethod
    def generate_cbn_from_template(v_topology, n_local_networks, n_vars_network, o_template, l_global_edges):
        """
        Generates a CBN (Coupled Boolean Network) using a given template and global edges.

        Args:
            v_topology: Topology of the CBN.
            n_local_networks (int): Number of local networks.
            n_vars_network (int): Number of variables per network.
            o_template: Template for local networks.
            l_global_edges (list): List of tuples representing the global edges between local networks.

        Returns:
            A CBN object generated from the provided template and global edges.
        """

        # Generate the local networks with indexes and variables (without relations or dynamics)
        l_local_networks = CBN.generate_local_networks_indexes_variables(n_local_networks=n_local_networks,
                                                                         n_vars_network=n_vars_network)

        # Generate the directed edges between the local networks
        l_directed_edges = []

        # Get the last index of the variables for the indexes of the directed edges
        i_last_variable = l_local_networks[-1].l_var_intern[-1] + 1

        # Generate the directed edges using the last variable generated and the selected output variables
        i_directed_edge = 1
        for relation in l_global_edges:
            output_local_network = relation[0]
            input_local_network = relation[1]

            # Get the output variables from the template
            l_output_variables = o_template.get_output_variables_from_template(output_local_network, l_local_networks)

            # Generate the coupling function
            coupling_function = " " + " ∨ ".join(list(map(str, l_output_variables))) + " "
            # Create the DirectedEdge object
            o_directed_edge = DirectedEdge(index=i_directed_edge,
                                           index_variable_signal=i_last_variable,
                                           input_local_network=input_local_network,
                                           output_local_network=output_local_network,
                                           l_output_variables=l_output_variables,
                                           coupling_function=coupling_function)
            i_last_variable += 1
            i_directed_edge += 1
            # Add the DirectedEdge object to the list
            l_directed_edges.append(o_directed_edge)

        # Process the coupling signals for every local network
        for o_local_network in l_local_networks:
            # Find the input signals for each local network
            l_input_signals = CBN.find_input_edges_by_network_index(index=o_local_network.index,
                                                                    l_directed_edges=l_directed_edges)
            # Process the input signals of the local network
            o_local_network.process_input_signals(l_input_signals=l_input_signals)

        # Generate the dynamics of the local networks using the template
        l_local_networks = CBN.generate_local_dynamic_with_template(o_template=o_template,
                                                                    l_local_networks=l_local_networks,
                                                                    l_directed_edges=l_directed_edges)

        # Generate the special Coupled Boolean Network (CBN)
        o_cbn = CBN(l_local_networks=l_local_networks,
                    l_directed_edges=l_directed_edges)

        # Add the Global Topology Object
        o_global_topology = GlobalTopology(v_topology=v_topology, l_edges=l_global_edges)
        o_cbn.o_global_topology = o_global_topology

        return o_cbn

    @staticmethod
    def generate_local_dynamic_with_template(o_template, l_local_networks, l_directed_edges):
        """
        Generates the dynamics for each local network using a given template and directed edges.

        Args:
            o_template: Template used to generate dynamics.
            l_local_networks (list): List of LocalNetwork objects to update.
            l_directed_edges (list): List of DirectedEdge objects for the connections between networks.

        Returns:
            list: Updated list of LocalNetwork objects with their dynamics.
        """
        number_max_of_clauses = 2  # Maximum number of clauses per variable (not used in this implementation)
        number_max_of_literals = 3  # Maximum number of literals per clause (not used in this implementation)

        # List to store the updated local networks
        l_local_networks_updated = []

        # Update the dynamics for each local network
        for o_local_network in l_local_networks:
            CustomText.print_simple_line()
            print("Local Network:", o_local_network.index)

            # List to hold the function descriptions for the variables
            des_funct_variables = []

            # Generate clauses for each local network based on the template
            for i_local_variable in o_local_network.l_var_intern:
                CustomText.print_simple_line()
                # Adapt the clause template to the specific variable
                l_clauses_node = CBN.update_clause_from_template(o_template=o_template,
                                                                 l_local_networks=l_local_networks,
                                                                 o_local_network=o_local_network,
                                                                 i_local_variable=i_local_variable,
                                                                 l_directed_edges=l_directed_edges)

                # Create an InternalVariable object with the generated clauses
                o_variable_model = InternalVariable(index=i_local_variable,
                                                    cnf_function=l_clauses_node)
                # Add the variable model to the list
                des_funct_variables.append(o_variable_model)

            # Update the local network with the function descriptions
            o_local_network.des_funct_variables = des_funct_variables.copy()
            l_local_networks_updated.append(o_local_network)
            print("Local network created:", o_local_network.index)
            CustomText.print_simple_line()

        # Return the updated list of local networks
        return l_local_networks_updated

    @staticmethod
    def update_clause_from_template(o_template, l_local_networks, o_local_network,
                                    i_local_variable, l_directed_edges):
        """
        Updates the clauses from the template for a specific variable in a local network.

        Args:
            o_template: The template containing CNF functions.
            l_local_networks (list): List of LocalNetwork objects.
            o_local_network: The specific LocalNetwork object being updated.
            i_local_variable: The index of the local variable to update.
            l_directed_edges (list): List of DirectedEdge objects.

        Returns:
            list: Updated list of clauses for the local variable.
        """

        # Gather all index variables from directed edges
        l_indexes_directed_edges = [o_directed_edge.index_variable for o_directed_edge in l_directed_edges]

        # Determine the CNF function index for the variable in the template
        n_local_variables = len(l_local_networks[0].l_var_intern)
        i_template_variable = i_local_variable - ((o_local_network.index - 1) * n_local_variables) + n_local_variables
        pre_l_clauses_node = o_template.d_variable_cnf_function[i_template_variable]

        print("Local Variable index:", i_local_variable)
        print("Template Variable index:", i_template_variable)
        print("Template Function:", pre_l_clauses_node)

        # Update the CNF function clauses with the specific variable index
        l_clauses_node = []
        for pre_clause in pre_l_clauses_node:
            # Create a clause by updating variable indices
            l_clause = []
            for template_value in pre_clause:
                # Determine the sign of the variable
                b_symbol = True
                if template_value < 0:
                    b_symbol = False

                # Update the variable index
                local_value = abs(template_value) + (
                            (o_local_network.index - 3) * n_local_variables) + n_local_variables

                # Check if the variable is internal or external
                if local_value not in o_local_network.l_var_intern:
                    # Use external variables if the local variable is not found
                    l_clause = o_local_network.l_var_exterm
                    break

                # Add the sign to the value
                if not b_symbol:
                    local_value = -local_value

                # Append the value to the clause
                l_clause.append(local_value)

            # Append the updated clause to the list
            l_clauses_node.append(l_clause)

        # Remove empty clauses
        l_clauses_node = [clause for clause in l_clauses_node if clause]

        print("Local Variable Index:", i_local_variable)
        print("CNF Function:", l_clauses_node)

        return l_clauses_node

    def get_kind_topology(self):
        return self.o_global_topology.v_topology

    def get_n_input_variables(self):
        pass

    def get_n_output_variables(self):
        pass
        
