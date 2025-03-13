import os
import sys
import time
import pandas as pd
import pickle
import copy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classes.localtemplates import LocalNetworkTemplate
from classes.utils.customtext import CustomText
from classes.globaltopology import GlobalTopology
from classes.cbnetwork import CBN

# Parámetros del experimento
N_SAMPLES = 10  # Número de muestras
START_EXPONENT = 3  # Comienza con 2^3 = 8 redes locales
MAX_EXPONENT = 5  # Máximo exponente: 2^6 = 64 redes locales
N_VARS_NETWORK = 5
N_OUTPUT_VARS = 2
N_INPUT_VARS = 2
V_TOPOLOGY = 2
N_MAX_CLAUSES = 2
N_MAX_LITERALS = 2

# Creación de directorios de salida
OUTPUT_FOLDER = 'outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
EXPERIMENT_NAME = "exp1_data_modified"
DIRECTORY_PATH = os.path.join(OUTPUT_FOLDER, f"{EXPERIMENT_NAME}_{2 ** START_EXPONENT}_{2 ** MAX_EXPONENT}_{N_SAMPLES}")
os.makedirs(DIRECTORY_PATH, exist_ok=True)
DIRECTORY_PKL = os.path.join(DIRECTORY_PATH, "pkl_cbn")
os.makedirs(DIRECTORY_PKL, exist_ok=True)
file_path = os.path.join(DIRECTORY_PATH, 'data.csv')

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Existing file deleted: {file_path}")

# Definición de los métodos para cada paso (solo se utilizan los pasos 1 y 2)
methods = {
    "find_local_attractors": {
        1: "find_local_attractors_sequential",
        2: "find_local_attractors_parallel",
        3: "find_local_attractors_parallel_with_weigths"
    },
    "find_compatible_pairs": {
        1: "find_compatible_pairs",
        2: "find_compatible_pairs_parallel",
        3: "find_compatible_pairs_parallel_with_weights"
    }
}

# Inicio del experimento
total_start_time = time.time()
print("BEGIN THE EXPERIMENT")
print("=" * 50)

for i_sample in range(1, N_SAMPLES + 1):
    o_template = LocalNetworkTemplate(
        n_vars_network=N_VARS_NETWORK,
        n_input_variables=N_INPUT_VARS,
        n_output_variables=N_OUTPUT_VARS,
        n_max_of_clauses=N_MAX_CLAUSES,
        n_max_of_literals=N_MAX_LITERALS,
        v_topology=V_TOPOLOGY
    )

    # Crear topología global inicial con 2^START_EXPONENT nodos
    initial_nodes = 2 ** START_EXPONENT
    o_global_topology = GlobalTopology.generate_sample_topology(v_topology=V_TOPOLOGY, n_nodes=initial_nodes)
    print("Generated Global Topology with", initial_nodes, "nodes")

    # Lista de tamaños de redes: potencias de 2 desde 2^START_EXPONENT hasta 2^MAX_EXPONENT
    network_sizes = [2 ** exp for exp in range(START_EXPONENT, MAX_EXPONENT + 1)]

    for n_local_networks in network_sizes:
        print(f"Experiment {i_sample} - Networks: {n_local_networks}, Variables: {N_VARS_NETWORK}")

        # Generar el objeto base CBN a partir de la plantilla
        base_cbn = CBN.generate_cbn_from_template(
            v_topology=V_TOPOLOGY,
            n_local_networks=n_local_networks,
            n_vars_network=N_VARS_NETWORK,
            o_template=o_template,
            l_global_edges=o_global_topology.l_edges
        )

        data_samples = []

        # Crear tres copias para cada variante
        sequential_instance = copy.deepcopy(base_cbn)
        parallel_instance = copy.deepcopy(base_cbn)
        weighted_instance = copy.deepcopy(base_cbn)

        # Mapeo de cada variante (1: secuencial, 2: paralelo, 3: paralelo con pesos) a su instancia y método correspondiente.
        variants = {
            1: (sequential_instance, {
                "find_local_attractors": methods["find_local_attractors"][1],
                "find_compatible_pairs": methods["find_compatible_pairs"][1]
            }),
            2: (parallel_instance, {
                "find_local_attractors": methods["find_local_attractors"][2],
                "find_compatible_pairs": methods["find_compatible_pairs"][2]
            }),
            3: (weighted_instance, {
                "find_local_attractors": methods["find_local_attractors"][3],
                "find_compatible_pairs": methods["find_compatible_pairs"][3]
            })
        }

        # Secuencia de pasos: solo se ejecutan los pasos 1 y 2
        step_names = ["find_local_attractors", "find_compatible_pairs"]

        # Para cada paso y variante, se ejecuta el método correspondiente y se registra el resultado.
        for step_index, step in enumerate(step_names, start=1):
            for variant in [1, 2, 3]:
                instance, method_mapping = variants[variant]
                method_name = method_mapping[step]
                try:
                    print(f"Executing {method_name} for step {step} (variant {variant})...")
                    start_time = time.perf_counter()
                    getattr(instance, method_name)()
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    print(f"{method_name} execution time: {execution_time:.6f} seconds")
                except Exception as e:
                    execution_time = None
                    print(f"Error in {method_name}: {e}")
                # Registrar resultados: para el paso 1 se registran los atractores locales y para el paso 2 los pares compatibles.
                sample_data = {
                    "i_sample": i_sample,
                    "n_local_networks": n_local_networks,
                    "n_var_network": N_VARS_NETWORK,
                    "v_topology": V_TOPOLOGY,
                    "n_output_variables": N_OUTPUT_VARS,
                    "n_clauses_function": N_MAX_CLAUSES,
                    "n_edges": n_local_networks,
                    "step": step_index,  # 1: atractores locales, 2: pares compatibles
                    "method": variant,  # 1: secuencial, 2: paralelo, 3: paralelo con pesos
                    "execution_time": execution_time,
                }
                if step_index == 1:
                    sample_data["n_local_attractors"] = instance.get_n_local_attractors()
                elif step_index == 2:
                    sample_data["n_pair_attractors"] = instance.get_n_pair_attractors()

                data_samples.append(sample_data)

        # Guardar resultados en CSV
        print("Data samples collected:", data_samples)
        df_results = pd.DataFrame(data_samples)

        mode = 'a' if os.path.exists(file_path) else 'w'
        header = not os.path.exists(file_path)
        df_results.to_csv(file_path, mode=mode, header=header, index=False)
        print(f"Experiment data saved in: {file_path}")

        # Guardar el objeto base CBN en un archivo pickle (para referencia)
        pickle_path = os.path.join(DIRECTORY_PKL, f'cbn_{i_sample}_{n_local_networks}.pkl')
        with open(pickle_path, 'wb') as file:
            pickle.dump(base_cbn, file)
        print(f"Pickle object saved in: {pickle_path}")

        # Actualizar la topología global: se añaden nodos hasta alcanzar el siguiente número requerido (si existe)
        if n_local_networks != network_sizes[-1]:
            # Calcular la diferencia entre el siguiente tamaño y el actual
            next_size = network_sizes[network_sizes.index(n_local_networks) + 1]
            nodes_to_add = next_size - n_local_networks
            for _ in range(nodes_to_add):
                o_global_topology.add_node()
            print(f"Added {nodes_to_add} nodes to global topology. New node count: {next_size}")

    CustomText.print_stars()
CustomText.print_dollars()

# Registrar tiempo total del experimento
total_end_time = time.time()
print(f"Total experiment time (seconds): {total_end_time - total_start_time}")
print("=" * 50)
print("END EXPERIMENT")
