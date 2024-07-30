# external imports
import os
import sys
import time
import pandas as pd
import pickle

from classes.cbnetwork import CBN

# Agregar el directorio principal del proyecto al sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# local imports
from classes.localtemplates import LocalNetworkTemplate
from classes.utils.customtext import CustomText
from classes.globaltopology import GlobalTopology

"""
Experiment 6 - Test the aleatory CBNs with different number of local networks
"""

# experiment parameters
N_SAMPLES = 1000
N_LOCAL_NETWORKS_MIN = 3
N_LOCAL_NETWORKS_MAX = 9
N_VARS_NETWORK = 5
N_OUTPUT_VARS = 2
N_INPUT_VARS = 2
V_TOPOLOGY = 2
N_MAX_CLAUSES = 2
N_MAX_LITERALS = 2
n_edges = 3

# verbose parameters
SHOW_MESSAGES = True

# Begin the Experiment
print("BEGIN THE EXPERIMENT")
print("=" * 80)

# Capture the time for all the experiment
v_begin_exp = time.time()

# Experiment Name
EXPERIMENT_NAME = "exp6_data"

# Create the 'outputs' directory if it doesn't exist
OUTPUT_FOLDER = 'outputs'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# create an experiment directory by parameters
DIRECTORY_PATH = (OUTPUT_FOLDER + "/" + EXPERIMENT_NAME + "_"
                  + str(N_LOCAL_NETWORKS_MIN) + "_"
                  + str(N_LOCAL_NETWORKS_MAX)
                  + "_" + str(N_SAMPLES))
os.makedirs(DIRECTORY_PATH, exist_ok=True)

# create a directory to save the pkl files
DIRECTORY_PKL = DIRECTORY_PATH + "/pkl_cbn"
os.makedirs(DIRECTORY_PKL, exist_ok=True)

# generate the experiment data file in csv
file_path = DIRECTORY_PATH + '/data.csv'

# Erase the file if exists
if os.path.exists(file_path):
    os.remove(file_path)
    print("Existing file deleted:", file_path)

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
                                                                n_nodes=N_LOCAL_NETWORKS_MIN,
                                                                n_edges=n_edges)

    for n_local_networks in range(N_LOCAL_NETWORKS_MIN, N_LOCAL_NETWORKS_MAX + 1):
        n_edges = n_local_networks
        l_data_sample = []
        print(f"EXPERIMENT {i_sample} OF {N_SAMPLES} TOPOLOGY: {V_TOPOLOGY}")
        print(f"NETWORKS: {n_local_networks} EDGES: {n_edges} VARIABLES: {N_VARS_NETWORK}")

        # GENERATE THE CBN WITH THE TOPOLOGY AND TEMPLATE
        o_cbn = CBN.generate_cbn_from_template(v_topology=V_TOPOLOGY,
                                               n_local_networks=n_local_networks,
                                               n_vars_network=N_VARS_NETWORK,
                                               o_template=o_template,
                                               l_global_edges=o_global_topology.l_edges)

        # find attractors
        v_begin_find_attractors = time.time()
        o_cbn.find_local_attractors_sequential()
        v_end_find_attractors = time.time()
        n_time_find_attractors = v_end_find_attractors - v_begin_find_attractors

        # find the compatible pairs
        v_begin_find_pairs = time.time()
        o_cbn.find_compatible_pairs()
        v_end_find_pairs = time.time()
        n_time_find_pairs = v_end_find_pairs - v_begin_find_pairs

        # Find attractor fields
        v_begin_find_fields = time.time()
        o_cbn.mount_stable_attractor_fields()
        v_end_find_fields = time.time()
        n_time_find_fields = v_end_find_fields - v_begin_find_fields

        # collect indicators
        d_collect_indicators = {
            # initial parameters
            "i_sample": i_sample,
            "n_local_networks": n_local_networks,
            "n_var_network": N_VAR_NETWORK,
            "v_topology": V_TOPOLOGY,
            "n_output_variables": N_OUTPUT_VARIABLES,
            "n_clauses_function": N_CLAUSES_FUNCTION,
            "n_edges": n_local_networks,
            # calculate parameters
            "n_local_attractors": o_cbn.get_n_local_attractors(),
            "n_pair_attractors": o_cbn.get_n_pair_attractors(),
            "n_attractor_fields": o_cbn.get_n_attractor_fields(),
            # time parameters
            "n_time_find_attractors": n_time_find_attractors,
            "n_time_find_pairs": n_time_find_pairs,
            "n_time_find_fields": n_time_find_fields
        }
        l_data_sample.append(d_collect_indicators)

        # save the collected indicator to profiler_analysis
        pf_res = pd.DataFrame(l_data_sample)
        pf_res.reset_index(drop=True, inplace=True)

        # if the file exist, open the 'a' mode (append), else create a new file
        mode = 'a' if os.path.exists(file_path) else 'w'
        # Add the header only if is a new file
        header = not os.path.exists(file_path)
        #  save the data in csv file
        pf_res.to_csv(file_path, mode=mode, header=header, index=False)

        print("Experiment data saved in:", file_path)

        # Open a file in binary write mode (wb)
        pickle_path = DIRECTORY_PKL + '/cbn_' + str(i_sample) + '_' + str(n_local_networks) + ".pkl"
        with open(pickle_path, 'wb') as file:
            # Use pickle.dump to save the object to the file
            pickle.dump(o_cbn, file)

        # Close the file
        file.close()
        print("Pickle object saved in:", pickle_path)

        # add node to topology
        o_global_topology.add_node()

        CustomText.print_duplex_line()
    CustomText.print_stars()
CustomText.print_dollars()

# Take the time of the experiment
v_end_exp = time.time()
v_time_exp = v_end_exp - v_begin_exp
print("Time experiment (in seconds): ", v_time_exp)

print("=" * 80)
print("END EXPERIMENT")
