import os
import time
import pandas as pd
import pickle

# Local imports
from classes.localtemplates import AleatoryTemplate
from classes.utils.customtext import CustomText

# Experiment parameters
N_SAMPLES = 1000
N_LOCAL_NETWORKS_MIN = 3
N_LOCAL_NETWORKS_MAX = 6
N_VAR_NETWORK = 5
N_OUTPUT_VARIABLES = 2
N_INPUT_VARIABLES = 2
V_TOPOLOGY = 2
N_CLAUSES_FUNCTION = 2

# Verbose parameters
SHOW_MESSAGES = True

# Begin the Experiment
print("BEGIN THE EXPERIMENT")
print("=" * 80)

# Capture the time for all the experiment
v_begin_exp = time.time()

# Experiment Name
EXPERIMENT_NAME = "exp7_data"

# Create the 'outputs' directory if it doesn't exist
OUTPUT_FOLDER = 'outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Create an experiment directory by parameters
DIRECTORY_PATH = (f"{OUTPUT_FOLDER}/{EXPERIMENT_NAME}_"
                  f"{N_LOCAL_NETWORKS_MIN}_{N_LOCAL_NETWORKS_MAX}_{N_SAMPLES}")
os.makedirs(DIRECTORY_PATH, exist_ok=True)

# Create a directory to save the pkl files
DIRECTORY_PKL = os.path.join(DIRECTORY_PATH, "pkl_cbn")
os.makedirs(DIRECTORY_PKL, exist_ok=True)

# Generate the experiment data file in csv
file_path = os.path.join(DIRECTORY_PATH, 'data.csv')

# Erase the file if it exists
if os.path.exists(file_path):
    os.remove(file_path)
    print("Existing file deleted:", file_path)

# Begin the process
for i_sample in range(1, N_SAMPLES + 1):
    # Generate the aleatory local network template object
    o_topology_template = AleatoryTemplate.generate_aleatory_template(n_var_network=N_VAR_NETWORK,
                                                                      v_topology=V_TOPOLOGY)
    for n_local_networks in range(N_LOCAL_NETWORKS_MIN, N_LOCAL_NETWORKS_MAX + 1):
        old_o_graph = None
        for n_edges in range(n_local_networks, (n_local_networks * 2) + 1):
            l_data_sample = []
            print(f"Experiment {i_sample} of {N_SAMPLES} TOPOLOGY: {V_TOPOLOGY} EDGES: {n_edges}")

            # If it is the first number of edges, generate a linear CBN from the template
            if old_o_graph is None:
                o_cbn = o_topology_template.generate_cbn_from_template(v_topology=V_TOPOLOGY,
                                                                       n_local_networks=n_local_networks,
                                                                       n_edges=n_edges)
            # Update the number of directed edges
            else:
                o_cbn = o_topology_template.generate_cbn_from_template(v_topology=V_TOPOLOGY,
                                                                       n_local_networks=n_local_networks,
                                                                       n_edges=n_edges,
                                                                       o_base_graph=old_o_graph)

            # Find attractors
            v_begin_find_attractors = time.time()
            o_cbn.find_local_attractors_sequential()
            v_end_find_attractors = time.time()
            n_time_find_attractors = v_end_find_attractors - v_begin_find_attractors

            # Find the compatible pairs
            v_begin_find_pairs = time.time()
            o_cbn.find_compatible_pairs()
            v_end_find_pairs = time.time()
            n_time_find_pairs = v_end_find_pairs - v_begin_find_pairs

            # Find attractor fields
            v_begin_find_fields = time.time()
            o_cbn.mount_stable_attractor_fields()
            v_end_find_fields = time.time()
            n_time_find_fields = v_end_find_fields - v_begin_find_fields

            # Collect indicators
            d_collect_indicators = {
                # Initial parameters
                "i_sample": i_sample,
                "n_local_networks": n_local_networks,
                "n_var_network": N_VAR_NETWORK,
                "v_topology": V_TOPOLOGY,
                "n_output_variables": N_OUTPUT_VARIABLES,
                "n_clauses_function": N_CLAUSES_FUNCTION,
                "n_edges": n_edges,
                # Calculate parameters
                "n_local_attractors": o_cbn.get_n_local_attractors(),
                "n_pair_attractors": o_cbn.get_n_pair_attractors(),
                "n_attractor_fields": o_cbn.get_n_attractor_fields(),
                # Time parameters
                "n_time_find_attractors": n_time_find_attractors,
                "n_time_find_pairs": n_time_find_pairs,
                "n_time_find_fields": n_time_find_fields
            }
            l_data_sample.append(d_collect_indicators)

            # Save the collected indicator to profiler_analysis
            pf_res = pd.DataFrame(l_data_sample)
            pf_res.reset_index(drop=True, inplace=True)

            # Save the data in CSV file
            mode = 'a' if os.path.exists(file_path) else 'w'
            header = not os.path.exists(file_path)
            pf_res.to_csv(file_path, mode=mode, header=header, index=False)

            print("Experiment data saved in:", file_path)

            # Save the object to a pickle file
            pickle_path = (f"{DIRECTORY_PKL}/cbn_{i_sample}_{n_local_networks}_{n_edges}.pkl")
            with open(pickle_path, 'wb') as file:
                pickle.dump(o_cbn, file)

            print("Pickle object saved in:", pickle_path)

            # Update old_graph
            old_o_graph = o_cbn.o_global_topology.o_graph

            CustomText.print_duplex_line()
    CustomText.print_stars()
CustomText.print_dollars()

# Take the time of the experiment
v_end_exp = time.time()
v_time_exp = v_end_exp - v_begin_exp
print("Time experiment (in seconds): ", v_time_exp)

print("=" * 80)
print("END EXPERIMENT")
