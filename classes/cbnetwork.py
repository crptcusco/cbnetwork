# external imports
import itertools  # libraries to iterate
import random
from itertools import product  # generate combinations of numbers
from typing import List, Optional, Any, Dict
from dask import delayed, compute   # Librarys to use Dask
import multiprocessing  # Library to use paralaleis woth process
from multiprocessing import Pool    # Library to generate a List of parallel
from math import ceil

# internal imports
from classes.globalscene import GlobalScene
from classes.globaltopology import GlobalTopology
from classes.internalvariable import InternalVariable
from classes.localnetwork import LocalNetwork
from classes.directededge import DirectedEdge
from classes.localscene import LocalAttractor
from classes.localtemplates import LocalNetworkTemplate
from classes.utils.customtext import CustomText

def process_local_network_mp(o_local_network):
    """
    Procesa una red local para encontrar sus atractores.
    Si hay un error, se imprime un mensaje y se devuelve la red sin cambios.
    """
    try:
        l_local_scenes = CBN._generate_local_scenes(o_local_network)
        updated_network = LocalNetwork.find_local_attractors(
            o_local_network=o_local_network,
            l_local_scenes=l_local_scenes
        )
        return updated_network  # ✅ Siempre retorna la red local actualizada
    except Exception as e:
        print(f"❌ ERROR en la red {o_local_network.id}: {e}")  # 📢 Registro del error
        return o_local_network  # ❗ Retorna la red sin cambios en caso de fallo

def process_output_signal_mp(args):
    """
    Procesa una señal de salida para calcular los pares compatibles.

    Args:
        args: Una tupla con:
              - signal_index: identificador de la señal.
              - l_attractors_input_0: lista de índices de atractores (valor 0).
              - l_attractors_input_1: lista de índices de atractores (valor 1).
              - index_variable: variable de la señal usada para obtener atractores.
              - get_attractors_func: función para obtener atractores (usualmente self.get_attractors_by_input_signal_value).

    Returns:
        Una tupla: (signal_index, d_comp_pairs_attractors_by_value, n_pairs)
    """
    signal_index, l_attractors_input_0, l_attractors_input_1, index_variable, get_attractors_func = args

    def find_attractor_pairs(signal_value, index_variable, l_attractors_input):
        l_attractors_output = [o_attractor.g_index for o_attractor in get_attractors_func(index_variable, signal_value)]
        return list(itertools.product(l_attractors_input, l_attractors_output))

    d_comp_pairs = {
        0: find_attractor_pairs(0, index_variable, l_attractors_input_0),
        1: find_attractor_pairs(1, index_variable, l_attractors_input_1)
    }
    n_pairs = len(d_comp_pairs[0]) + len(d_comp_pairs[1])
    return signal_index, d_comp_pairs, n_pairs

def evaluate_pair(base_pairs: list, candidate_pair: tuple, d_local_attractors) -> bool:
    """
    Comprueba si un par candidato es compatible con los pares base.

    Aplanando recursivamente cada par para obtener índices individuales.
    """

    def flatten(x):
        if isinstance(x, (list, tuple)):
            for item in x:
                yield from flatten(item)
        else:
            yield x

    # Aplanamos la estructura de cada par base para obtener todos los índices individuales.
    base_attractor_indices = {x for pair in base_pairs for x in flatten(pair)}

    # Para cada índice obtenido, se extrae el primer elemento del valor correspondiente en d_local_attractors.
    already_visited_networks = {d_local_attractors[idx][0] for idx in base_attractor_indices}

    double_check = 0
    # Aplanamos también el par candidato para iterar sobre cada índice individual.
    for candidate_idx in flatten(candidate_pair):
        if d_local_attractors[candidate_idx][0] in already_visited_networks:
            if candidate_idx in base_attractor_indices:
                double_check += 1
        else:
            double_check += 1

    return double_check == 2

def cartesian_product_mod(base_pairs: list, candidate_pairs: list, d_local_attractors) -> list:
    """
    Realiza el producto cartesiano modificado entre dos listas de pares, filtrando las combinaciones incompatibles.
    """
    field_pair_list = []
    for base_pair in base_pairs:
        for candidate_pair in candidate_pairs:
            # Si base_pair es una tupla, lo convertimos a lista para concatenar
            if isinstance(base_pair, tuple):
                base_pair = list(base_pair)
            if evaluate_pair(base_pair, candidate_pair, d_local_attractors):
                new_pair = base_pair + [candidate_pair]
                field_pair_list.append(new_pair)
    return field_pair_list

def _convert_to_tuple(x):
    """
    Convierte recursivamente listas en tuplas para que sean hashables.
    """
    if isinstance(x, list):
        return tuple(_convert_to_tuple(item) for item in x)
    return x

def process_single_base_pair(base_pair, candidate_pairs, d_local_attractors):
    """
    Procesa un único par base, aplicando el producto cartesiano modificado
    entre [base_pair] y candidate_pairs, utilizando el diccionario d_local_attractors.

    Args:
        base_pair: Un único par base (puede ser una tupla o lista).
        candidate_pairs (list): Lista de pares candidatos.
        d_local_attractors: Diccionario de atractores locales.

    Returns:
        list: Lista de nuevos pares resultantes.
    """
    # Llamamos a cartesian_product_mod pasando [base_pair] para que trabaje con un único elemento.
    return cartesian_product_mod([base_pair], candidate_pairs, d_local_attractors)


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
        self.d_global_scenes_count ={}

        # Graphs
        self.o_global_topology = None

    # PRINCIPAL FUNCTIONS
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
                # print(edge)

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
                # print(f"Local Network with index {o_local_network_update.index} updated")
                return True

        # If no network was found, print an error message
        # print(f"ERROR: Local Network with index {o_local_network_update.index} not found")
        return False

    @staticmethod
    def _generate_local_scenes(o_local_network: LocalNetwork) -> Optional[List[str]]:
        """
        Generate local scenes for the given local network based on external variables.

        Args:
            o_local_network (LocalNetwork): The local network object.

        Returns:
            Optional[List[str]]: A list of local scenes or None if there are no external variables.
        """
        # if len(o_local_network.l_var_exterm) != 0:
        #     return list(product('01', repeat=len(o_local_network.l_var_exterm)))
        # return None

        external_vars_count = len(o_local_network.l_var_exterm)
        if external_vars_count > 0:
            # Generate binary combinations for the external variables
            return [''.join(scene) for scene in product('01', repeat=external_vars_count)]
        return None

    def find_local_attractors_sequential(self) -> None:
        """
        Finds local attractors sequentially and updates the list of local attractors in the object.

        This method calculates the local attractors for each local network, updates the coupling signals,
        assigns global indices to each attractor, and generates the attractor dictionary.
        """
        CustomText.make_title('FIND LOCAL ATTRACTORS')

        for o_local_network in self.l_local_networks:
            # Generate the local network scenes
            l_local_scenes = CBN._generate_local_scenes(o_local_network)
            # Calculate the local attractors for the local network
            o_local_network = LocalNetwork.find_local_attractors(
                o_local_network=o_local_network,
                l_local_scenes=l_local_scenes
            )

        # Update the coupling signals to be analyzed
        for o_local_network in self.l_local_networks:
            self.process_kind_signal(o_local_network)

        # Assign global indices to each attractor
        self._assign_global_indices_to_attractors()

        # Generate the attractor dictionary
        self.generate_attractor_dictionary()

        print('Number of local attractors:', self._count_total_attractors())
        CustomText.make_sub_sub_title('END FIND LOCAL ATTRACTORS')

    def find_local_attractors_parallel(self, num_cpus=None):
        """
        Paraleliza el proceso de encontrar atractores locales utilizando multiprocessing.

        Para cada red local en self.l_local_networks se:
          1. Generan sus escenas locales.
          2. Se calculan sus atractores locales.

        Luego, en el proceso principal se actualiza cada red con su respectivo procesamiento
        de señales, se asignan los índices globales a cada atractor y se genera el diccionario de atractores.

        Esta función es equivalente a la versión secuencial, pero distribuye el cálculo de cada
        red local en procesos independientes.
        """
        CustomText.make_title("FIND LOCAL ATTRACTORS PARALLEL")

        if num_cpus is None:
            num_cpus = multiprocessing.cpu_count()

        # Crear un pool de procesos; se puede ajustar el número de procesos si es necesario
        with multiprocessing.Pool(processes=num_cpus) as pool:
            # map() enviará cada elemento de self.l_local_networks a la función process_local_network_mp
            updated_networks = pool.map(process_local_network_mp, self.l_local_networks)

        # Actualizar la lista de redes locales con los resultados obtenidos
        self.l_local_networks = list(updated_networks)

        # Asignar índices globales a cada atractor
        self._assign_global_indices_to_attractors()

        # Generar el diccionario de atractores
        self.generate_attractor_dictionary()

        CustomText.make_sub_sub_title("END FIND LOCAL ATTRACTORS PARALLEL")

    def find_local_attractors_parallel_with_weigths(self, num_cpus=None):
        """
        Encuentra atractores locales en paralelo con multiprocessing, balanceando la carga
        mediante un sistema de 'buckets' según el peso de cada tarea.
        """
        CustomText.make_title("FIND LOCAL ATTRACTORS WEIGHTED BALANCED")

        # Crear lista de tareas con peso
        tasks_with_weight = []
        for o_local_network in self.l_local_networks:
            num_vars = len(o_local_network.l_var_total) if hasattr(o_local_network, 'l_var_total') else 0
            num_coupling = len(o_local_network.l_input_signals) if hasattr(o_local_network, 'l_input_signals') else 0
            weight = num_vars * (2 ** num_coupling)
            tasks_with_weight.append((weight, o_local_network))

        # Ordenar por peso descendente
        tasks_with_weight.sort(key=lambda x: x[0], reverse=True)

        # Crear buckets balanceados
        buckets = [{'total': 0, 'tasks': []} for _ in range(num_cpus)]
        for weight, task in tasks_with_weight:
            bucket = min(buckets, key=lambda b: b['total'])
            bucket['tasks'].append(task)
            bucket['total'] += weight

        # Imprimir info inicial
        print(f"\nNúmero de workers: {num_cpus}")
        print("Distribución de tareas por bucket antes de la ejecución:")
        for i, bucket in enumerate(buckets):
            print(f"  Bucket {i}: {len(bucket['tasks'])} tasks, total weight: {bucket['total']}")

        # Ejecutar en paralelo con multiprocessing
        all_tasks = [task for bucket in buckets for task in bucket['tasks']]
        with Pool(processes=num_cpus) as pool:
            results = pool.map(process_local_network_mp, all_tasks)

        # Verificar si alguna red desapareció
        if len(results) != len(self.l_local_networks):
            print(f"⚠️ ERROR: Se perdieron {len(self.l_local_networks) - len(results)} redes en el proceso!")

        # Emparejar redes originales con los resultados usando índices
        ordered_results = [None] * len(self.l_local_networks)
        for original, processed in zip(all_tasks, results):
            index = self.l_local_networks.index(original)  # Buscar la posición original
            ordered_results[index] = processed

        # Verificar si hubo algún None (indica error en la asignación)
        if None in ordered_results:
            print(f"⚠️ ERROR: Algunos resultados no se reasignaron correctamente!")

        self.l_local_networks = ordered_results  # Asignar en el orden correcto

        # Procesar señales por red local
        for o_local_network in self.l_local_networks:
            self.process_kind_signal(o_local_network)

        # Asignar índices globales a los attratores
        self._assign_global_indices_to_attractors()
        # Generar el diccionario de attractores
        self.generate_attractor_dictionary()

        # Imprimir info final de los buckets
        print("\nInformación final de los buckets:")
        for i, bucket in enumerate(buckets):
            print(f"  Bucket {i}: {len(bucket['tasks'])} tasks, total weight: {bucket['total']}")

        CustomText.make_sub_sub_title("END FIND LOCAL ATTRACTORS WEIGHTED BALANCED")

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
                # print("INFO: the output signal is restricted")
            elif signal_set_length == 2:
                o_output_signal.kind_signal = 3
                # print("INFO: the output signal is stable")
            else:
                o_output_signal.kind_signal = 4
                # print("INFO: the scene signal is not stable. This CBN doesn't have stable Attractor Fields")

    def _count_total_attractors(self) -> int:
        """
        Count the total number of attractors across all local networks.

        Returns:
            int: The total number of attractors.
        """
        return sum(len(o_local_scene.l_attractors) for o_local_network in self.l_local_networks for o_local_scene in
                   o_local_network.l_local_scenes)

    def find_compatible_pairs(self) -> None:
        """
        Generate pairs of attractors using the output signal.

        Returns:
            None: Updates the dictionary of compatible attractor pairs in the object.
        """

        CustomText.make_title('FIND COMPATIBLE ATTRACTOR PAIRS')

        # Procesar señales de acoplamiento para cada red local
        for o_local_network in self.l_local_networks:
            self.process_kind_signal(o_local_network)

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

        # print(f'Number of attractor pairs: {n_pairs}')
        CustomText.make_sub_sub_title('END FIND ATTRACTOR PAIRS')

    def find_compatible_pairs_parallel(self, num_cpus=None):
        """
        Paraleliza la generación de pares compatibles utilizando multiprocessing,
        procesando cada señal de salida en un proceso independiente.

        Para cada red local, se obtienen sus output edges y, para cada señal de salida:
          - Se extraen las listas de atractores de entrada (para 0 y 1).
          - Se procesa la señal para calcular los pares compatibles mediante la función auxiliar.

        Al finalizar, se actualizan los objetos de señal con los diccionarios de pares y se
        imprime el total de pares encontrados.

        Actualiza el estado interno del objeto (self.l_local_networks) con las señales modificadas.
        """
        CustomText.make_title("FIND COMPATIBLE ATTRACTOR PAIRS (PARALLEL)")

        # Procesar señales de acoplamiento para cada red local
        for o_local_network in self.l_local_networks:
            self.process_kind_signal(o_local_network)

        if num_cpus is None:
            num_cpus = multiprocessing.cpu_count()

        tasks = []
        signal_map = {}
        # Recorrer todas las redes locales
        for o_local_network in self.l_local_networks:
            l_output_edges = self.get_output_edges_by_network_index(o_local_network.index)
            # Procesar cada output signal
            for o_output_signal in l_output_edges:
                signal_index = o_output_signal.index
                signal_map[signal_index] = o_output_signal  # Guardar referencia para actualizar luego
                l_attractors_input_0 = [attr.g_index for attr in o_output_signal.d_out_value_to_attractor[0]]
                l_attractors_input_1 = [attr.g_index for attr in o_output_signal.d_out_value_to_attractor[1]]
                task_args = (signal_index, l_attractors_input_0, l_attractors_input_1,
                             o_output_signal.index_variable, self.get_attractors_by_input_signal_value)
                tasks.append(task_args)

        print(f"Tareas creadas: {len(tasks)}")

        # Ejecutar las tareas en paralelo
        with Pool(processes=num_cpus) as pool:
            results = pool.map(process_output_signal_mp, tasks)

        print(f"Resultados obtenidos: {len(results)}")
        total_pairs = 0
        # Actualizar los objetos de salida con los resultados obtenidos
        for signal_index, d_comp_pairs, n_signal_pairs in results:
            if signal_index not in signal_map:
                print(f"Error: Índice de señal {signal_index} no encontrado en signal_map")
                continue
            o_output_signal = signal_map[signal_index]
            o_output_signal.d_comp_pairs_attractors_by_value = d_comp_pairs
            total_pairs += n_signal_pairs

        CustomText.make_sub_sub_title(f"END FIND COMPATIBLE ATTRACTOR PAIRS (Total pairs: {total_pairs})")

    def find_compatible_pairs_parallel_with_weights(self, num_cpus=None):
        """
        Paraleliza la generación de pares compatibles utilizando multiprocessing,
        asignando las tareas (cada una correspondiente a una señal de acoplamiento)
        a buckets balanceados por peso. El peso de cada tarea se calcula como:

             weight = len(l_attractors_input_0) + len(l_attractors_input_1)

        Luego, todas las tareas se ejecutan en paralelo y se actualizan los objetos originales.
        """
        CustomText.make_title("FIND COMPATIBLE ATTRACTOR PAIRS (PARALLEL WITH WEIGHTS)")

        # Procesar señales de acoplamiento para cada red local
        for o_local_network in self.l_local_networks:
            self.process_kind_signal(o_local_network)

        if num_cpus is None:
            num_cpus = multiprocessing.cpu_count()

        tasks_with_weight = []
        signal_map = {}

        # Recorrer cada red local y sus señales de salida
        for o_local_network in self.l_local_networks:
            l_output_edges = self.get_output_edges_by_network_index(o_local_network.index)
            for o_output_signal in l_output_edges:
                signal_index = o_output_signal.index
                # Guardar la referencia para actualizar luego
                signal_map[signal_index] = o_output_signal
                l_attractors_input_0 = [attr.g_index for attr in o_output_signal.d_out_value_to_attractor[0]]
                l_attractors_input_1 = [attr.g_index for attr in o_output_signal.d_out_value_to_attractor[1]]
                # Definir el peso de la tarea (puedes ajustar esta fórmula si lo deseas)
                weight = len(l_attractors_input_0) + len(l_attractors_input_1)
                task_args = (signal_index, l_attractors_input_0, l_attractors_input_1,
                             o_output_signal.index_variable, self.get_attractors_by_input_signal_value)
                tasks_with_weight.append((weight, task_args))

        # Ordenar las tareas por peso de mayor a menor
        tasks_with_weight.sort(key=lambda x: x[0], reverse=True)

        # Crear buckets para balancear la carga entre los CPUs
        buckets = [{'total': 0, 'tasks': []} for _ in range(num_cpus)]
        for weight, task in tasks_with_weight:
            bucket = min(buckets, key=lambda b: b['total'])
            bucket['tasks'].append(task)
            bucket['total'] += weight

        # Imprimir información de los buckets antes de ejecutar
        print(f"\nNúmero de CPUs: {num_cpus}")
        print("Distribución de tareas por bucket antes de la ejecución:")
        for i, bucket in enumerate(buckets):
            print(f"  Bucket {i}: {len(bucket['tasks'])} tasks, total weight: {bucket['total']}")

        # Combinar todas las tareas en una sola lista para la ejecución paralela
        all_tasks = []
        for bucket in buckets:
            all_tasks.extend(bucket['tasks'])

        # Ejecutar todas las tareas en paralelo utilizando multiprocessing
        with Pool(processes=num_cpus) as pool:
            results = pool.map(process_output_signal_mp, all_tasks)

        print(f"\nNúmero de tareas procesadas: {len(results)}")

        total_pairs = 0
        # Actualizar los objetos de señal con los resultados
        for signal_index, d_comp_pairs, n_signal_pairs in results:
            if signal_index not in signal_map:
                print(f"Error: Índice de señal {signal_index} no encontrado en signal_map")
                continue
            o_output_signal = signal_map[signal_index]
            o_output_signal.d_comp_pairs_attractors_by_value = d_comp_pairs
            total_pairs += n_signal_pairs

        print(f"Total de pares de atractores: {total_pairs}")

        # Imprimir la información final de los buckets (no se modifican en la ejecución, solo informativos)
        print("\nInformación final de los buckets:")
        for i, bucket in enumerate(buckets):
            print(f"  Bucket {i}: {len(bucket['tasks'])} tasks, total weight: {bucket['total']}")

        CustomText.make_sub_sub_title("END FIND COMPATIBLE ATTRACTOR PAIRS (PARALLEL WITH WEIGHTS)")

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
        # print("Directed Edges ordered.")

    def order_edges_by_grade(self):
        """
        Orders the directed edges based on the total degree of the networks they connect.

        The total degree of an edge is the sum of the degrees of its input and output networks.
        The edges are then reordered to keep adjacent edges (sharing networks) close together.
        """

        # Paso 1: Calcular el grado de cada red local
        network_degrees = {net.index: 0 for net in self.l_local_networks}

        for edge in self.l_directed_edges:
            network_degrees[edge.input_local_network] += 1
            network_degrees[edge.output_local_network] += 1

        # Paso 2: Calcular el "grado total" de cada arista
        def calculate_edge_grade(edge):
            input_degree = network_degrees.get(edge.input_local_network, 0)
            output_degree = network_degrees.get(edge.output_local_network, 0)
            return input_degree + output_degree

        # Paso 3: Ordenar aristas por el grado total en orden descendente
        self.l_directed_edges.sort(key=calculate_edge_grade, reverse=True)

        # Paso 4: Reordenar para mantener aristas adyacentes juntas
        def is_adjacent(edge1, edge2):
            return (edge1.input_local_network == edge2.input_local_network or
                    edge1.input_local_network == edge2.output_local_network or
                    edge1.output_local_network == edge2.input_local_network or
                    edge1.output_local_network == edge2.output_local_network)

        ordered_edges = [self.l_directed_edges.pop(0)]  # Comenzamos con la arista de mayor grado

        while self.l_directed_edges:
            for i, edge in enumerate(self.l_directed_edges):
                if is_adjacent(ordered_edges[-1], edge):
                    ordered_edges.append(self.l_directed_edges.pop(i))
                    break
            else:
                # Si no encontramos adyacente, agregamos la siguiente disponible
                ordered_edges.append(self.l_directed_edges.pop(0))

        # Paso 5: Actualizar la lista de aristas
        self.l_directed_edges = ordered_edges

    def disorder_edges(self):
        """
        Desordena aleatoriamente la lista de aristas dirigidas, asegurando que la primera
        arista no tenga vértices en común con la segunda, y reasigna las aristas en la estructura.
        """
        if len(self.l_directed_edges) < 2:
            return  # No hay suficiente número de aristas para aplicar la condición

        # Mezclar aleatoriamente las aristas
        random.shuffle(self.l_directed_edges)

        # Verificar si la primera y la segunda comparten algún vértice
        def have_common_vertex(edge1, edge2):
            return (edge1.input_local_network in {edge2.input_local_network, edge2.output_local_network} or
                    edge1.output_local_network in {edge2.input_local_network, edge2.output_local_network})

        # Si la primera y segunda arista tienen un vértice en común, buscar una nueva segunda arista
        if have_common_vertex(self.l_directed_edges[0], self.l_directed_edges[1]):
            for i in range(2, len(self.l_directed_edges)):
                if not have_common_vertex(self.l_directed_edges[0], self.l_directed_edges[i]):
                    # Intercambiar la segunda arista con una que no comparta vértices
                    self.l_directed_edges[1], self.l_directed_edges[i] = self.l_directed_edges[i], \
                    self.l_directed_edges[1]
                    break

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

        # print("Number of attractor fields found:", len(l_base_pairs))
        CustomText.make_sub_sub_title("END MOUNT ATTRACTOR FIELDS")


    def mount_stable_attractor_fields_parallel(self, num_cpus=None):
        """
        Ensambla campos de atractores estables en paralelo utilizando multiprocessing.

        El proceso es:
          1. Se ordenan las aristas por compatibilidad.
          2. Se genera la base inicial de pares a partir de la primera arista.
          3. Para cada arista restante (a partir de la segunda):
                 - Se extrae la lista de pares candidatos de esa señal de salida.
                 - Para cada elemento de la base actual (cada par base), se crea un proceso que aplica\n               cartesian_product_mod() con ese elemento y los pares candidatos.\n             - Se esperan todos los procesos y se unen los resultados (se convierten a tuplas para que sean hashables),\n               actualizando así la base de pares.\n      4. Finalmente, se genera el diccionario de campos de atractores a partir de la base final.\n\n    Actualiza self.d_attractor_fields con los campos encontrados.
        """
        CustomText.make_title("MOUNT STABLE ATTRACTOR FIELDS (PARALLEL)")

        if num_cpus is None:
            num_cpus = multiprocessing.cpu_count()

        # Paso 1: Ordenar las aristas por compatibilidad
        self.order_edges_by_compatibility()

        # Paso 2: Generar la base inicial de pares a partir de la primera arista
        l_base_pairs = set(self.l_directed_edges[0].d_comp_pairs_attractors_by_value[0] +
                           self.l_directed_edges[0].d_comp_pairs_attractors_by_value[1])

        # Paso 3: Iterar sobre las aristas restantes para refinar la base de pares
        for o_directed_edge in self.l_directed_edges[1:]:
            # Extraer la lista de pares candidatos para la señal de salida actual
            l_candidate_pairs = (o_directed_edge.d_comp_pairs_attractors_by_value[0] +
                                 o_directed_edge.d_comp_pairs_attractors_by_value[1])
            print(f"\nProcesando arista {o_directed_edge.index} con {len(l_base_pairs)} pares base")

            if not l_base_pairs:
                break

            # Convertir la base actual en lista para iterar individualmente
            base_pairs_list = list(l_base_pairs)

            # Preparar argumentos para cada tarea: cada base_pair se procesa individualmente
            tasks_args = [(bp, l_candidate_pairs, self.d_local_attractors) for bp in base_pairs_list]

            with Pool(processes=num_cpus) as pool:
                results = pool.starmap(process_single_base_pair, tasks_args)

            # Cada resultado es una lista (posiblemente vacía) de nuevos pares a partir de un base_pair.
            # Unir todos los nuevos pares en un único conjunto (convertidos a tuplas para ser hashables).
            new_base_pairs = set()
            for r in results:
                # r es una lista de nuevos pares; convertimos cada nuevo par a tupla (si no lo es) y lo agregamos al conjunto.
                new_base_pairs.update({tuple(item) if isinstance(item, list) else item for item in r})

            l_base_pairs = new_base_pairs
            print(f"Base actualizada: {len(l_base_pairs)} pares")
            if not l_base_pairs:
                break

        # Paso 4: Generar el diccionario de campos de atractores a partir de la base final
        self.d_attractor_fields = {}
        for i, base_element in enumerate(l_base_pairs, start=1):
            # Aseguramos que cada base_element sea una estructura iterable (convertida a tupla si es necesario)
            if not isinstance(base_element, (list, tuple)):
                raise TypeError(f"Esperado lista o tupla, pero se recibió {type(base_element)}: {base_element}")
            # Para cada campo, se extraen los elementos de cada par y se eliminan duplicados
            self.d_attractor_fields[i] = list(
                {tuple(item) if isinstance(item, list) else item for pair in base_element for item in pair})

        CustomText.make_sub_sub_title("END MOUNT STABLE ATTRACTOR FIELDS (PARALLEL)")

    def mount_stable_attractor_fields_parallel_chunks(self, num_cpus=None):
        """
        Ensambla campos de atractores estables en paralelo utilizando multiprocessing.

        El proceso es:
          1. Se ordenan las aristas por compatibilidad.
          2. Se genera la base inicial de pares a partir de la primera arista.
          3. Para cada arista restante (a partir de la segunda):
               - Se extraen los pares candidatos de esa señal de salida.
               - Se divide la base actual en chunks de tamaño uniforme (según num_cpus).
               - Se procesa en paralelo cada chunk mediante cartesian_product_mod,
                 pasando como candidatos la lista extraída y el diccionario d_local_attractors.
               - Se unen los resultados (por unión de conjuntos) para actualizar la base de pares.
          4. Finalmente, se genera el diccionario de campos de atractores a partir de la base final.

        Actualiza self.d_attractor_fields con los campos encontrados.
        """
        CustomText.make_title("MOUNT STABLE ATTRACTOR FIELDS (PARALLEL CHUNKS)")

        if num_cpus is None:
            num_cpus = multiprocessing.cpu_count()

        # Paso 1: Ordenar las aristas por compatibilidad
        self.order_edges_by_compatibility()

        # Paso 2: Generar la base inicial de pares a partir de la primera arista
        l_base_pairs = set(self.l_directed_edges[0].d_comp_pairs_attractors_by_value[0] +
                           self.l_directed_edges[0].d_comp_pairs_attractors_by_value[1])

        # Paso 3: Iterar sobre las aristas restantes para refinar la base de pares
        for o_directed_edge in self.l_directed_edges[1:]:
            # Extraer la lista de pares candidatos para la señal de salida actual
            l_candidate_pairs = (o_directed_edge.d_comp_pairs_attractors_by_value[0] +
                                 o_directed_edge.d_comp_pairs_attractors_by_value[1])

            # Dividir la base actual en chunks de tamaño uniforme
            l_base_pairs_list = list(l_base_pairs)
            n = len(l_base_pairs_list)
            if n == 0:
                break
            chunk_size = ceil(n / num_cpus)
            chunks = [l_base_pairs_list[i:i + chunk_size] for i in range(0, n, chunk_size)]

            print(f"\nProcesando arista {o_directed_edge.index} con {n} pares base; chunk size: {chunk_size}")
            for i, chunk in enumerate(chunks):
                print(f"  Chunk {i}: {len(chunk)} pares")

            # Ejecutar en paralelo: para cada chunk, llamar a cartesian_product_mod
            candidate_pairs = l_candidate_pairs  # Candidatos para esta iteración
            with Pool(processes=num_cpus) as pool:
                args = [(chunk, candidate_pairs, self.d_local_attractors) for chunk in chunks]
                iter_results = pool.starmap(cartesian_product_mod, args)

            # Unir los resultados: cada resultado es una lista de nuevos pares
            new_base_pairs = set()
            for r in iter_results:
                # Convertir cada par (lista) a tupla para poder hacer la unión en el set
                new_base_pairs = new_base_pairs.union({tuple(item) if isinstance(item, list) else item for item in r})
            l_base_pairs = new_base_pairs

            print(f"Base actualizada: {len(l_base_pairs)} pares")
            if not l_base_pairs:
                break

        # Paso 4: Generar el diccionario de campos de atractores a partir de la base final
        self.d_attractor_fields = {}
        for i, base_element in enumerate(l_base_pairs, start=1):

            if not isinstance(base_element, (list, tuple)):
                raise TypeError(f"Esperado lista ou tupla, mas recebeu {type(base_element)}: {base_element}")

            # Si base_element es una lista de pares, convertimos cada par a tupla y eliminamos duplicados
            self.d_attractor_fields[i] = list(
                {tuple(item) if isinstance(item, list) else item for pair in base_element for item in pair})

        CustomText.make_sub_sub_title("END MOUNT STABLE ATTRACTOR FIELDS (PARALLEL CHUNKS)")

    # DASK FUNCTIONS
    def dask_find_local_attractors(self):
        """
        Paraleliza el proceso de encontrar atractores locales utilizando Dask.

        Esta función divide el cálculo de atractores locales en subtareas paralelas y luego combina los resultados.
        """
        CustomText.make_title("FIND LOCAL ATTRACTORS")

        # Paso 1: Crear tareas paralelas para encontrar atractores locales
        def process_local_network(o_local_network):
            """
            Procesa una red local: genera escenas locales, encuentra atractores, y procesa señales.
            """
            # Generar escenas locales
            l_local_scenes = CBN._generate_local_scenes(o_local_network)

            # Encontrar atractores locales
            updated_network = LocalNetwork.find_local_attractors(
                o_local_network=o_local_network,
                l_local_scenes=l_local_scenes
            )

            return updated_network

        # Crear una lista de tareas usando dask.delayed
        delayed_tasks = [delayed(process_local_network)(o_local_network) for o_local_network in self.l_local_networks]

        # Ejecutar todas las tareas en paralelo
        updated_networks = compute(*delayed_tasks)

        # Actualizar las redes locales con los resultados
        self.l_local_networks = list(updated_networks)  # Convertimos la tupla a lista para mantener el formato original

        # Procesar señales de acoplamiento
        for o_local_network in self.l_local_networks:
            self.process_kind_signal(o_local_network)

        # Paso 2: Asignar índices globales a cada atractor
        self._assign_global_indices_to_attractors()

        # Paso 3: Generar el diccionario de atractores
        self.generate_attractor_dictionary()

        CustomText.make_sub_sub_title("END FIND LOCAL ATTRACTORS")

    def dask_find_local_attractors_weighted_balanced(self, num_workers):
        """
        Paraleliza el proceso de encontrar atractores locales utilizando Dask,
        alocando las tareas según un peso definido como:

             peso = (cantidad de variables) * 2^(número de señales de acoplamiento)

        Luego, las tareas se agrupan en 'num_workers' buckets de forma que el peso
        total de cada bucket sea lo más equilibrado posible. Finalmente, se programan
        todas las tareas al mismo tiempo para que se ejecuten concurrentemente, y se
        actualiza la estructura del CBN.
        """
        CustomText.make_title("FIND LOCAL ATTRACTORS WEIGHTED BALANCED")

        # Función que se ejecutará para cada red local
        def process_local_network(o_local_network):
            # Genera las escenas locales usando el método (o función) estático de CBN
            l_local_scenes = CBN._generate_local_scenes(o_local_network)
            # Encuentra los atractores locales para la red (se asume que este método actualiza internamente el objeto)
            updated_network = LocalNetwork.find_local_attractors(
                o_local_network=o_local_network,
                l_local_scenes=l_local_scenes
            )
            return updated_network

        # Crear una lista de tareas junto con su peso
        tasks_with_weight = []
        for o_local_network in self.l_local_networks:
            # Se asume que cada red local tiene:
            #  - l_var_total: lista de variables (internas/externas/totales)
            #  - l_input_signals: lista de señales de acoplamiento (o atributo similar)
            num_vars = len(o_local_network.l_var_total) if hasattr(o_local_network, 'l_var_total') else 0
            num_coupling = len(o_local_network.l_input_signals) if hasattr(o_local_network, 'l_input_signals') else 0
            weight = num_vars * (2 ** num_coupling)
            delayed_task = delayed(process_local_network)(o_local_network)
            tasks_with_weight.append((weight, delayed_task))

        # Ordenar las tareas por peso en orden descendente
        tasks_with_weight.sort(key=lambda x: x[0], reverse=True)

        # Crear buckets (grupos) para cada worker, para balancear la carga
        buckets = [{'total': 0, 'tasks': []} for _ in range(num_workers)]
        for weight, task in tasks_with_weight:
            # Asignar la tarea al bucket con menor peso acumulado
            bucket = min(buckets, key=lambda b: b['total'])
            bucket['tasks'].append(task)
            bucket['total'] += weight

        # Para depuración, imprimir los pesos acumulados de cada bucket
        for i, bucket in enumerate(buckets):
            print(f"Bucket {i} total weight: {bucket['total']} with {len(bucket['tasks'])} tasks")

        # En lugar de computar cada bucket secuencialmente, combinamos todas las tareas
        all_tasks = []
        for bucket in buckets:
            all_tasks.extend(bucket['tasks'])

        # Ejecutar todas las tareas al mismo tiempo
        results = compute(*all_tasks)

        # Actualizar la lista de redes locales con los resultados combinados
        self.l_local_networks = list(results)

        # Procesar señales de acoplamiento para cada red local (paso adicional en el flujo)
        for o_local_network in self.l_local_networks:
            self.process_kind_signal(o_local_network)

        # Paso 2: Asignar índices globales a cada atractor
        self._assign_global_indices_to_attractors()

        # Paso 3: Generar el diccionario de atractores
        self.generate_attractor_dictionary()

        CustomText.make_sub_sub_title("END FIND LOCAL ATTRACTORS WEIGHTED BALANCED")

    def dask_find_compatible_pairs(self) -> None:
        """
        Paraleliza la generación de pares de atractores usando señales de salida.

        Utiliza Dask para calcular pares compatibles y asegura que los resultados
        se integren correctamente a los objetos originales.
        """
        CustomText.make_title("FIND COMPATIBLE ATTRACTOR PAIRS")

        # Función auxiliar para encontrar pares de atractores
        def find_attractor_pairs(signal_value, o_output_signal_index_variable, l_attractors_input):
            """
            Encuentra pares de atractores basados en el valor de la señal de entrada.

            Args:
                signal_value (int): El valor de la señal (0 o 1).
                o_output_signal_index_variable: Indice de la variable del objeto de señal de salida.
                l_attractors_input (list): Lista de índices de atractores para la señal de entrada.

            Returns:
                list: Lista de pares de atractores.
            """
            l_attractors_output = [
                o_attractor.g_index
                for o_attractor in
                self.get_attractors_by_input_signal_value(o_output_signal_index_variable, signal_value)
            ]
            return list(itertools.product(l_attractors_input, l_attractors_output))

        # Función auxiliar para procesar una señal de salida
        def process_output_signal(signal_index, l_attractors_input_0, l_attractors_input_1, index_variable):
            """
            Procesa una señal de salida y encuentra pares compatibles.

            Args:
                signal_index: Índice de la señal de salida.
                l_attractors_input_0: Lista de atractores para valor 0.
                l_attractors_input_1: Lista de atractores para valor 1.
                index_variable: Variable de índice de la señal.

            Returns:
                dict: Diccionario con los pares de atractores.
            """
            d_comp_pairs_attractors_by_value = {
                0: find_attractor_pairs(0, index_variable, l_attractors_input_0),
                1: find_attractor_pairs(1, index_variable, l_attractors_input_1),
            }

            # Retorna el índice de la señal y el diccionario generado
            n_pairs = len(d_comp_pairs_attractors_by_value[0]) + len(d_comp_pairs_attractors_by_value[1])
            return signal_index, d_comp_pairs_attractors_by_value, n_pairs

        # Crear una lista de tareas paralelas
        delayed_tasks = []
        signal_map = {}
        for o_local_network in self.l_local_networks:
            l_output_edges = self.get_output_edges_by_network_index(o_local_network.index)
            for o_output_signal in l_output_edges:
                signal_index = o_output_signal.index
                signal_map[signal_index] = o_output_signal  # Mapeo para acceder a los objetos originales
                l_attractors_input_0 = [attr.g_index for attr in o_output_signal.d_out_value_to_attractor[0]]
                l_attractors_input_1 = [attr.g_index for attr in o_output_signal.d_out_value_to_attractor[1]]
                delayed_tasks.append(
                    delayed(process_output_signal)(
                        signal_index, l_attractors_input_0, l_attractors_input_1, o_output_signal.index_variable
                    )
                )

        # Antes de computar tareas
        print(f"Tareas creadas: {len(delayed_tasks)}")
        for task in delayed_tasks[:5]:  # Muestra solo las primeras 5
            print(task)

        # Ejecutar las tareas en paralelo
        results = compute(*delayed_tasks)

        # Después de ejecutar compute
        print(f"Resultados obtenidos: {len(results)}")
        for result in results[:5]:
            print(result)

        for idx, result in enumerate(results[:5]):  # Mostrar solo los primeros 5 para evitar demasiada información
            print(f"Resultado {idx}: {result}")

        # Actualizar los objetos originales con los resultados
        n_pairs = 0
        for signal_index, d_comp_pairs_attractors_by_value, n_signal_pairs in results:

            if signal_index not in signal_map:
                print(f"Error: Índice de señal {signal_index} no encontrado en signal_map")
                continue  # Saltar este resultado si hay un problema

            o_output_signal = signal_map[signal_index]
            o_output_signal.d_comp_pairs_attractors_by_value = d_comp_pairs_attractors_by_value
            n_pairs += n_signal_pairs

        # Mostrar el resultado final
        # print(f"Number of attractor pairs: {n_pairs}")
        CustomText.make_sub_sub_title("END FIND ATTRACTOR PAIRS")

    # SHOW FUNCTIONS
    def show_directed_edges(self) -> None:
        CustomText.print_duplex_line()
        print("SHOW THE DIRECTED EDGES OF THE CBN")
        for o_directed_edge in self.l_directed_edges:
            o_directed_edge.show()

    def show_directed_edges_order(self) -> None:
        CustomText.print_duplex_line()
        print("SHOW THE EDGES", end=" ")
        print(" ".join(f"{o_directed_edge.index}: {o_directed_edge.get_edge()}"
                       for o_directed_edge in self.l_directed_edges))

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

    def show_local_attractors_dictionary(self) -> None:
        # Método para mostrar el diccionario de atractores locales
        CustomText.make_title('Global Dictionary of local attractors')
        for key, value in self.d_local_attractors.items():
            print(key, '->', value)

    def show_stable_attractor_fields_detailed(self) -> None:
        # Método para mostrar campos de atractores estables de manera detallada
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

    def show_attractors_fields(self) -> None:
        # Método para mostrar campos de atractores
        CustomText.make_sub_title('List of attractor fields')
        for key, value in self.d_attractor_fields.items():
            print(key, "->", value)
        print(f"Number of attractor fields found: {len(self.d_attractor_fields)}")

    # GENERATE FUNCTIONS
    def generate_global_scenes(self) -> None:
        # Método para generar escenas globales
        CustomText.make_title('Generated Global Scenes')
        l_edges_indexes = [o_directed_edge.index_variable for o_directed_edge in self.l_directed_edges]
        binary_combinations = list(product([0, 1], repeat=len(l_edges_indexes)))
        self.l_global_scenes = [GlobalScene(l_edges_indexes, list(combination)) for combination in binary_combinations]
        CustomText.make_sub_title('Global Scenes generated')

    def plot_topology(self, ax=None) -> None:
        # Método para graficar la topología
        self.o_global_topology.plot_topology(ax=ax)

    def count_fields_by_global_scenes(self):
        # Método para contar campos estables por escenas globales
        # CustomText.make_sub_title('Counting the stable attractor fields by global scene')
        self.d_global_scenes_count: Dict[str, int] = {}
        for key, o_attractor_field in self.d_attractor_fields.items():
            d_variable_value = {}
            for i_attractor in o_attractor_field:
                o_attractor = self.get_local_attractor_by_index(i_attractor)
                if o_attractor:
                    for aux_pos, aux_variable in enumerate(o_attractor.relation_index):
                        d_variable_value[aux_variable] = o_attractor.local_scene[aux_pos]
            sorted_dict = {k: d_variable_value[k] for k in sorted(d_variable_value)}
            combination_key = ''.join(str(sorted_dict[k]) for k in sorted_dict)
            if combination_key in self.d_global_scenes_count:
                self.d_global_scenes_count[combination_key] += 1
            else:
                self.d_global_scenes_count[combination_key] = 1
        # return self.d_global_scenes_count

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
        o_template = LocalNetworkTemplate(n_vars_network=n_vars_network, n_input_variables=n_input_variables,
                                          n_output_variables=n_output_variables, n_max_of_clauses=n_max_of_clauses,
                                          n_max_of_literals=n_max_of_literals, v_topology=v_topology)

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
    def find_output_edges_by_network_index(index: int, l_directed_edges: List['DirectedEdge'] ) -> List['DirectedEdge']:
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
            # CustomText.print_simple_line()
            # print("Local Network:", o_local_network.index)

            # List to hold the function descriptions for the variables
            des_funct_variables = []

            # Generate clauses for each local network based on the template
            for i_local_variable in o_local_network.l_var_intern:
                CustomText.print_simple_line()
                # Adapt the clause template to the 5_specific variable
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
            # print("Local network created:", o_local_network.index)
            # CustomText.print_simple_line()

        # Return the updated list of local networks
        return l_local_networks_updated

    @staticmethod
    def update_clause_from_template(o_template, l_local_networks, o_local_network,i_local_variable, l_directed_edges):
        """
        Updates the clauses from the template for a 5_specific variable in a local network.

        Args:
            o_template: The template containing CNF functions.
            l_local_networks (list): List of LocalNetwork objects.
            o_local_network: The 5_specific LocalNetwork object being updated.
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

        # print("Local Variable index:", i_local_variable)
        # print("Template Variable index:", i_template_variable)
        # print("Template Function:", pre_l_clauses_node)

        # Update the CNF function clauses with the 5_specific variable index
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

        # print("Local Variable Index:", i_local_variable)
        # print("CNF Function:", l_clauses_node)

        return l_clauses_node

    # GETTERS FUNCTIONS
    def get_local_attractor_by_index(self, i_attractor: int) -> Optional[LocalAttractor]:
        # Método para obtener un atractor local por su índice
        for o_local_network in self.l_local_networks:
            for o_scene in o_local_network.l_local_scenes:
                for o_attractor in o_scene.l_attractors:
                    if o_attractor.g_index == i_attractor:
                        return o_attractor
        # print('ERROR: Attractor index not found')
        return None

    def get_kind_topology(self):
        return self.o_global_topology.v_topology

    def get_n_input_variables(self):
        pass

    def get_n_output_variables(self):
        pass
        
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

    def get_global_scene_attractor_fields(self):
        return self.d_global_scenes_count
