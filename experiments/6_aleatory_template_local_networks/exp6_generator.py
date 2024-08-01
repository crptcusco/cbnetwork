# External imports
import os
import sys
import time
import pandas as pd
import pickle

# Add the project's main directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local imports
from classes.localtemplates import LocalNetworkTemplate
from classes.utils.customtext import CustomText
from classes.globaltopology import GlobalTopology
from classes.cbnetwork import CBN

"""
Experiment 6 - Test the aleatory CBNs with different number of local networks
"""

# Experiment parameters
N_SAMPLES = 100
N_LOCAL_NETWORKS_MIN = 3
N_LOCAL_NETWORKS_MAX = 9
N_VARS_NETWORK = 5
N_OUTPUT_VARS = 2
N_INPUT_VARS = 2
V_TOPOLOGY = 2
N_MAX_CLAUSES = 2
N_MAX_LITERALS = 2

# Verbose parameters
SHOW_MESSAGES = True

# Begin the Experiment
print("BEGIN THE EXPERIMENT")
print("=" * 50)

# Capture the time for the entire experiment
v_begin_exp = time.time()

# Experiment Name
EXPERIMENT_NAME = "exp6_data"

# Create the 'outputs' directory if it doesn't exist
OUTPUT_FOLDER = 'outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Create an experiment directory by parameters
DIRECTORY_PATH = os.path.join(OUTPUT_FOLDER,
                              f"{EXPERIMENT_NAME}_{N_LOCAL_NETWORKS_MIN}_{N_LOCAL_NETWORKS_MAX}_{N_SAMPLES}")
os.makedirs(DIRECTORY_PATH, exist_ok=True)

# Create a directory to save the pickle files
DIRECTORY_PKL = os.path.join(DIRECTORY_PATH, "pkl_cbn")
os.makedirs(DIRECTORY_PKL, exist_ok=True)

# Generate the experiment data file in CSV
file_path = os.path.join(DIRECTORY_PATH, 'data.csv')

# Erase the file if it exists
if os.path.exists(file_path):
    os.remove(file_path)
    print(f"Existing file deleted: {file_path}")

# Begin the process
for i_sample in range(1, N_SAMPLES + 1):  # 1 - 1000 , 1, 2

    # GENERATE THE LOCAL NETWORK TEMPLATE
    o_template = LocalNetworkTemplate(v_topology=V_TOPOLOGY,
                                      n_vars_network=N_VARS_NETWORK,
                                      n_input_variables=N_INPUT_VARS,
                                      n_output_variables=N_OUTPUT_VARS,
                                      n_max_of_clauses=N_MAX_CLAUSES,
                                      n_max_of_literals=N_MAX_LITERALS)

    # GENERATE THE GLOBAL TOPOLOGY
    o_global_topology = GlobalTopology.generate_sample_topology(v_topology=V_TOPOLOGY,
                                                                n_nodes=N_LOCAL_NETWORKS_MIN)
    print("Generated Global Topology")

    for n_local_networks in range(N_LOCAL_NETWORKS_MIN, N_LOCAL_NETWORKS_MAX + 1):
        l_data_sample = []
        print(f"Experiment {i_sample} of {N_SAMPLES} - Topology: {V_TOPOLOGY}")
        print(f"Networks: {n_local_networks} Variables: {N_VARS_NETWORK}")

        # GENERATE THE CBN WITH THE TOPOLOGY AND TEMPLATE
        o_cbn = CBN.generate_cbn_from_template(v_topology=V_TOPOLOGY,
                                               n_local_networks=n_local_networks,
                                               n_vars_network=N_VARS_NETWORK,
                                               o_template=o_template,
                                               l_global_edges=o_global_topology.l_edges)

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
            "n_var_network": N_VARS_NETWORK,
            "v_topology": V_TOPOLOGY,
            "n_output_variables": N_OUTPUT_VARS,
            "n_clauses_function": N_MAX_CLAUSES,
            "n_edges": n_local_networks,
            # Calculated parameters
            "n_local_attractors": o_cbn.get_n_local_attractors(),
            "n_pair_attractors": o_cbn.get_n_pair_attractors(),
            "n_attractor_fields": o_cbn.get_n_attractor_fields(),
            # Time parameters
            "n_time_find_attractors": n_time_find_attractors,
            "n_time_find_pairs": n_time_find_pairs,
            "n_time_find_fields": n_time_find_fields
        }
        l_data_sample.append(d_collect_indicators)

        # Save the collected indicators to CSV
        pf_res = pd.DataFrame(l_data_sample)
        pf_res.reset_index(drop=True, inplace=True)

        mode = 'a' if os.path.exists(file_path) else 'w'
        header = not os.path.exists(file_path)
        pf_res.to_csv(file_path, mode=mode, header=header, index=False)

        print(f"Experiment data saved in: {file_path}")

        # Save the object to a pickle file
        pickle_path = os.path.join(DIRECTORY_PKL, f'cbn_{i_sample}_{n_local_networks}.pkl')
        with open(pickle_path, 'wb') as file:
            pickle.dump(o_cbn, file)

        print(f"Pickle object saved in: {pickle_path}")

        # Add node to topology
        o_global_topology.add_node()

        CustomText.print_duplex_line()
    CustomText.print_stars()
CustomText.print_dollars()

# Take the time of the experiment
v_end_exp = time.time()
v_time_exp = v_end_exp - v_begin_exp
print(f"Time experiment (in seconds): {v_time_exp}")

print("=" * 50)
print("END EXPERIMENT")
