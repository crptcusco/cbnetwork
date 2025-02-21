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

# Experiment parameters
N_SAMPLES = 10
N_LOCAL_NETWORKS_MIN = 3
N_LOCAL_NETWORKS_MAX = 10
N_VARS_NETWORK = 5
N_OUTPUT_VARS = 2
N_INPUT_VARS = 2
V_TOPOLOGY = 2
N_MAX_CLAUSES = 2
N_MAX_LITERALS = 2

# Create output directories
OUTPUT_FOLDER = 'outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
EXPERIMENT_NAME = "exp1_data"
DIRECTORY_PATH = os.path.join(OUTPUT_FOLDER,
                              f"{EXPERIMENT_NAME}_{N_LOCAL_NETWORKS_MIN}_{N_LOCAL_NETWORKS_MAX}_{N_SAMPLES}")
os.makedirs(DIRECTORY_PATH, exist_ok=True)
DIRECTORY_PKL = os.path.join(DIRECTORY_PATH, "pkl_cbn")
os.makedirs(DIRECTORY_PKL, exist_ok=True)
file_path = os.path.join(DIRECTORY_PATH, 'data.csv')

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Existing file deleted: {file_path}")

# Method groups for each step
methods = {
    "find_local_attractors": [
        "find_local_attractors_sequential",
        "find_local_attractors_parallel",
        "find_local_attractors_parallel_with_weigths"
    ],
    "find_compatible_pairs": [
        "find_compatible_pairs",
        "find_compatible_pairs_parallel",
        "find_compatible_pairs_parallel_with_weights"
    ],
    "mount_stable_attractor_fields": [
        "mount_stable_attractor_fields",
        "mount_stable_attractor_fields_parallel",
        "mount_stable_attractor_fields_parallel_chunks"
    ]
}

# Begin experiment
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

    o_global_topology = GlobalTopology.generate_sample_topology(v_topology=V_TOPOLOGY, n_nodes=N_LOCAL_NETWORKS_MIN)
    print("Generated Global Topology")

    for n_local_networks in range(N_LOCAL_NETWORKS_MIN, N_LOCAL_NETWORKS_MAX + 1):
        print(f"Experiment {i_sample} - Networks: {n_local_networks}, Variables: {N_VARS_NETWORK}")

        # Generate the base CBN object
        base_cbn = CBN.generate_cbn_from_template(
            v_topology=V_TOPOLOGY,
            n_local_networks=n_local_networks,
            n_vars_network=N_VARS_NETWORK,
            o_template=o_template,
            l_global_edges=o_global_topology.l_edges
        )

        data_samples = []

        for step_index, (step, method_list) in enumerate(methods.items(), start=1):
            for method_index, method_name in enumerate(method_list, start=1):
                cbn_instance = copy.deepcopy(base_cbn)  # Ensure each method runs independently

                try:
                    print(f"Executing {method_name}...")
                    start_time = time.perf_counter()
                    getattr(cbn_instance, method_name)()
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    print(f"{method_name} execution time: {execution_time:.6f} seconds")
                except Exception as e:
                    execution_time = None  # Indica que hubo un error
                    print(f"Error in {method_name}: {e}")


                # Collect results
                data_samples.append({
                    "i_sample": i_sample,
                    "n_local_networks": n_local_networks,
                    "n_var_network": N_VARS_NETWORK,
                    "v_topology": V_TOPOLOGY,
                    "n_output_variables": N_OUTPUT_VARS,
                    "n_clauses_function": N_MAX_CLAUSES,
                    "n_edges": n_local_networks,
                    "step": step_index,
                    "method": method_index,
                    "execution_time": execution_time,
                    "n_local_attractors": cbn_instance.get_n_local_attractors() if step_index == 1 else None,
                    "n_pair_attractors": cbn_instance.get_n_pair_attractors() if step_index == 2 else None,
                    "n_attractor_fields": cbn_instance.get_n_attractor_fields() if step_index == 3 else None
                })

        # Save results to CSV
        df_results = pd.DataFrame(data_samples)
        mode = 'a' if os.path.exists(file_path) else 'w'
        header = not os.path.exists(file_path)
        df_results.to_csv(file_path, mode=mode, header=header, index=False)

        print(f"Experiment data saved in: {file_path}")
        pickle_path = os.path.join(DIRECTORY_PKL, f'cbn_{i_sample}_{n_local_networks}.pkl')
        with open(pickle_path, 'wb') as file:
            pickle.dump(base_cbn, file)

        print(f"Pickle object saved in: {pickle_path}")
        o_global_topology.add_node()

    CustomText.print_stars()
CustomText.print_dollars()

# Record total experiment time
total_end_time = time.time()
print(f"Total experiment time (seconds): {total_end_time - total_start_time}")
print("=" * 50)
print("END EXPERIMENT")
