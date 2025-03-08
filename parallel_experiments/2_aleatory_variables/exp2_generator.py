import os
import sys
import time
import pandas as pd
import pickle
import copy

# Agregar el directorio padre al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classes.localtemplates import LocalNetworkTemplate
from classes.utils.customtext import CustomText
from classes.globaltopology import GlobalTopology
from classes.cbnetwork import CBN

# Parámetros del experimento
N_SAMPLES = 1000
N_LOCAL_NETWORKS = 8
VARS_NETWORK_LIST = [5, 6, 7, 8, 9, 10]  # Número de variables por red
N_OUTPUT_VARS = 2
N_INPUT_VARS = 2
V_TOPOLOGY = 2
N_MAX_CLAUSES = 2
N_MAX_LITERALS = 2

# Directorios de salida
OUTPUT_FOLDER = 'outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
EXPERIMENT_NAME = "exp2_data"
DIRECTORY_PATH = os.path.join(OUTPUT_FOLDER, f"{EXPERIMENT_NAME}_{N_LOCAL_NETWORKS}_{N_SAMPLES}")
os.makedirs(DIRECTORY_PATH, exist_ok=True)
DIRECTORY_PKL = os.path.join(DIRECTORY_PATH, "pkl_cbn")
os.makedirs(DIRECTORY_PKL, exist_ok=True)
file_path = os.path.join(DIRECTORY_PATH, 'data.csv')

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Existing file deleted: {file_path}")

# Métodos para cada paso
# Variante 1: secuencial; Variante 2: paralela; Variante 3: paralela ponderada (con chunks)
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
    },
    "mount_stable_attractor_fields": {
        1: "mount_stable_attractor_fields",
        2: "mount_stable_attractor_fields_parallel",
        3: "mount_stable_attractor_fields_parallel_chunks"
    }
}

total_start_time = time.time()
print("BEGIN THE EXPERIMENT")
print("=" * 50)

for i_sample in range(1, N_SAMPLES + 1):
    # Se itera sobre el número de variables para cada red
    for n_vars_network in VARS_NETWORK_LIST:
        # Crear template con el número de variables actual
        o_template = LocalNetworkTemplate(
            n_vars_network=n_vars_network,
            n_input_variables=N_INPUT_VARS,
            n_output_variables=N_OUTPUT_VARS,
            n_max_of_clauses=N_MAX_CLAUSES,
            n_max_of_literals=N_MAX_LITERALS,
            v_topology=V_TOPOLOGY
        )

        # Generar topología global con 6 redes locales
        o_global_topology = GlobalTopology.generate_sample_topology(v_topology=V_TOPOLOGY, n_nodes=N_LOCAL_NETWORKS)
        print("Generated Global Topology")
        print(f"Experiment {i_sample} - Networks: {N_LOCAL_NETWORKS}, Variables: {n_vars_network}")

        # Generar el objeto base CBN a partir del template y la topología global
        base_cbn = CBN.generate_cbn_from_template(
            v_topology=V_TOPOLOGY,
            n_local_networks=N_LOCAL_NETWORKS,
            n_vars_network=n_vars_network,
            o_template=o_template,
            l_global_edges=o_global_topology.l_edges
        )

        data_samples = []

        # Crear tres copias para las tres variantes:
        sequential_instance = copy.deepcopy(base_cbn)
        parallel_instance = copy.deepcopy(base_cbn)
        weighted_instance = copy.deepcopy(base_cbn)

        # Mapear cada variante (1: secuencial, 2: paralela, 3: paralela ponderada) a su instancia y métodos correspondientes.
        variants = {
            1: (sequential_instance, {
                "find_local_attractors": methods["find_local_attractors"][1],
                "find_compatible_pairs": methods["find_compatible_pairs"][1],
                "mount_stable_attractor_fields": methods["mount_stable_attractor_fields"][1]
            }),
            2: (parallel_instance, {
                "find_local_attractors": methods["find_local_attractors"][2],
                "find_compatible_pairs": methods["find_compatible_pairs"][2],
                "mount_stable_attractor_fields": methods["mount_stable_attractor_fields"][2]
            }),
            3: (weighted_instance, {
                "find_local_attractors": methods["find_local_attractors"][3],
                "find_compatible_pairs": methods["find_compatible_pairs"][3],
                "mount_stable_attractor_fields": methods["mount_stable_attractor_fields"][3]
            })
        }

        # Secuencia de pasos
        step_names = ["find_local_attractors", "find_compatible_pairs", "mount_stable_attractor_fields"]

        # Para cada paso y para cada variante, ejecutar el método correspondiente y registrar el resultado.
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
                # Registrar resultados
                data_samples.append({
                    "i_sample": i_sample,
                    "n_local_networks": N_LOCAL_NETWORKS,
                    "n_var_network": n_vars_network,
                    "v_topology": V_TOPOLOGY,
                    "n_output_variables": N_OUTPUT_VARS,
                    "n_clauses_function": N_MAX_CLAUSES,
                    "n_edges": N_LOCAL_NETWORKS,
                    "step": step_index,  # 1: attractores locales, 2: pares compatibles, 3: campos de attractores
                    "method": variant,  # 1: secuencial, 2: paralela, 3: paralela ponderada
                    "execution_time": execution_time,
                    "n_local_attractors": instance.get_n_local_attractors() if step_index == 1 else None,
                    "n_pair_attractors": instance.get_n_pair_attractors() if step_index == 2 else None,
                    "n_attractor_fields": instance.get_n_attractor_fields() if step_index == 3 else None
                })

        # Guardar resultados en CSV
        print("Data samples collected:", data_samples)
        df_results = pd.DataFrame(data_samples)

        mode = 'a' if os.path.exists(file_path) else 'w'
        header = not os.path.exists(file_path)
        df_results.to_csv(file_path, mode=mode, header=header, index=False)
        print(f"Experiment data saved in: {file_path}")

        # Guardar el objeto base CBN en un archivo pickle con el sufijo del número de variables
        pickle_filename = f'cbn_{i_sample}_{N_LOCAL_NETWORKS}_{n_vars_network}.pkl'
        pickle_path = os.path.join(DIRECTORY_PKL, pickle_filename)
        with open(pickle_path, 'wb') as file:
            pickle.dump(base_cbn, file)
        print(f"Pickle object saved in: {pickle_path}")

    # Imprimir separador al finalizar cada muestra
    CustomText.print_stars()

CustomText.print_dollars()

# Registro del tiempo total del experimento
total_end_time = time.time()
print(f"Total experiment time (seconds): {total_end_time - total_start_time}")
print("=" * 50)
print("END EXPERIMENT")
