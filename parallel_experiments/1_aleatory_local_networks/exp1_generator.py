import copy
import os
import pickle
import sys
import time

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classes.cbnetwork import CBN
from classes.globaltopology import GlobalTopology
from classes.localtemplates import LocalNetworkTemplate
from classes.utils.customtext import CustomText

# Experiment parameters
N_SAMPLES = 1000
N_LOCAL_NETWORKS_MIN = 3
N_LOCAL_NETWORKS_MAX = 8
N_VARS_NETWORK = 5
N_OUTPUT_VARS = 2
N_INPUT_VARS = 2
V_TOPOLOGY = 2
N_MAX_CLAUSES = 2
N_MAX_LITERALS = 2

# Create output directories
OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
EXPERIMENT_NAME = "exp1_data"
DIRECTORY_PATH = os.path.join(
    OUTPUT_FOLDER,
    f"{EXPERIMENT_NAME}_{N_LOCAL_NETWORKS_MIN}_{N_LOCAL_NETWORKS_MAX}_{N_SAMPLES}",
)
os.makedirs(DIRECTORY_PATH, exist_ok=True)
DIRECTORY_PKL = os.path.join(DIRECTORY_PATH, "pkl_cbn")
os.makedirs(DIRECTORY_PKL, exist_ok=True)
file_path = os.path.join(DIRECTORY_PATH, "data.csv")

if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Existing file deleted: {file_path}")

# Method groups for each step
# Variant 1: sequential; Variant 2: parallel; Variant 3: weighted parallel (with chunks)
methods = {
    "find_local_attractors": {
        1: "find_local_attractors_sequential",
        2: "find_local_attractors_parallel",
        3: "find_local_attractors_parallel_with_weigths",
    },
    "find_compatible_pairs": {
        1: "find_compatible_pairs",
        2: "find_compatible_pairs_parallel",
        3: "find_compatible_pairs_parallel_with_weights",
    },
    "mount_stable_attractor_fields": {
        1: "mount_stable_attractor_fields",
        2: "mount_stable_attractor_fields_parallel",
        3: "mount_stable_attractor_fields_parallel_chunks",
    },
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
        v_topology=V_TOPOLOGY,
    )

    o_global_topology = GlobalTopology.generate_sample_topology(
        v_topology=V_TOPOLOGY, n_nodes=N_LOCAL_NETWORKS_MIN
    )
    print("Generated Global Topology")

    for n_local_networks in range(N_LOCAL_NETWORKS_MIN, N_LOCAL_NETWORKS_MAX + 1):
        print(
            f"Experiment {i_sample} - Networks: {n_local_networks}, Variables: {N_VARS_NETWORK}"
        )

        # Generate the base CBN object
        base_cbn = CBN.generate_cbn_from_template(
            v_topology=V_TOPOLOGY,
            n_local_networks=n_local_networks,
            n_vars_network=N_VARS_NETWORK,
            o_template=o_template,
            l_global_edges=o_global_topology.l_edges,
        )

        data_samples = []

        # Create three copies for the three variants:
        sequential_instance = copy.deepcopy(base_cbn)
        parallel_instance = copy.deepcopy(base_cbn)
        weighted_instance = copy.deepcopy(base_cbn)

        # Map each variant (1: sequential, 2: parallel, 3: weighted) to its instance and corresponding method names.
        variants = {
            1: (
                sequential_instance,
                {
                    "find_local_attractors": methods["find_local_attractors"][1],
                    "find_compatible_pairs": methods["find_compatible_pairs"][1],
                    "mount_stable_attractor_fields": methods[
                        "mount_stable_attractor_fields"
                    ][1],
                },
            ),
            2: (
                parallel_instance,
                {
                    "find_local_attractors": methods["find_local_attractors"][2],
                    "find_compatible_pairs": methods["find_compatible_pairs"][2],
                    "mount_stable_attractor_fields": methods[
                        "mount_stable_attractor_fields"
                    ][2],
                },
            ),
            3: (
                weighted_instance,
                {
                    "find_local_attractors": methods["find_local_attractors"][3],
                    "find_compatible_pairs": methods["find_compatible_pairs"][3],
                    "mount_stable_attractor_fields": methods[
                        "mount_stable_attractor_fields"
                    ][3],
                },
            ),
        }

        # Define the sequence of steps
        step_names = [
            "find_local_attractors",
            "find_compatible_pairs",
            "mount_stable_attractor_fields",
        ]

        # For each step and for each variant, execute the corresponding method and record the result.
        for step_index, step in enumerate(step_names, start=1):
            for variant in [1, 2, 3]:
                instance, method_mapping = variants[variant]
                method_name = method_mapping[step]
                try:
                    print(
                        f"Executing {method_name} for step {step} (variant {variant})..."
                    )
                    start_time = time.perf_counter()
                    getattr(instance, method_name)()
                    end_time = time.perf_counter()
                    execution_time = end_time - start_time
                    print(f"{method_name} execution time: {execution_time:.6f} seconds")
                except Exception as e:
                    execution_time = None
                    print(f"Error in {method_name}: {e}")
                # Collect results regardless of whether an error occurred.
                data_samples.append(
                    {
                        "i_sample": i_sample,
                        "n_local_networks": n_local_networks,
                        "n_var_network": N_VARS_NETWORK,
                        "v_topology": V_TOPOLOGY,
                        "n_output_variables": N_OUTPUT_VARS,
                        "n_clauses_function": N_MAX_CLAUSES,
                        "n_edges": n_local_networks,
                        "step": step_index,  # 1: local attractors, 2: pair attractors, 3: attractor fields
                        "method": variant,  # 1: sequential, 2: parallel, 3: weighted parallel
                        "execution_time": execution_time,
                        "n_local_attractors": (
                            instance.get_n_local_attractors()
                            if step_index == 1
                            else None
                        ),
                        "n_pair_attractors": (
                            instance.get_n_pair_attractors()
                            if step_index == 2
                            else None
                        ),
                        "n_attractor_fields": (
                            instance.get_n_attractor_fields()
                            if step_index == 3
                            else None
                        ),
                    }
                )

        # Save results to CSV
        print("Data samples collected:", data_samples)
        df_results = pd.DataFrame(data_samples)

        mode = "a" if os.path.exists(file_path) else "w"
        header = not os.path.exists(file_path)
        df_results.to_csv(file_path, mode=mode, header=header, index=False)
        print(f"Experiment data saved in: {file_path}")

        # Save the original base instance to a pickle file (for reference)
        pickle_path = os.path.join(
            DIRECTORY_PKL, f"cbn_{i_sample}_{n_local_networks}.pkl"
        )
        with open(pickle_path, "wb") as file:
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
