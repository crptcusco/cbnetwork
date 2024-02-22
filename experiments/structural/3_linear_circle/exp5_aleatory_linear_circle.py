# external imports
import os
import time
import pandas as pd
from memory_profiler import profile
import pickle

# local imports
from PathCircleTemplate import PathCircleTemplate
from classes.utils.customtext import CustomText

"""
Experiment 5 - Test the path and 2_ring structures 
using aleatory generated template for the local network 
"""


@profile
def run_script():
    # experiment parameters
    N_SAMPLES = 10
    N_LOCAL_NETWORKS_MIN = 12
    N_LOCAL_NETWORKS_MAX = 12
    N_VAR_NETWORK = 5
    N_OUTPUT_VARIABLES = 2
    N_INPUT_VARIABLES = 2
    # V_TOPOLOGY = 3  # cycle graph
    N_CLAUSES_FUNCTION = 2
    N_DIRECTED_EDGES = 1

    # verbose parameters
    SHOW_MESSAGES = True

    # Begin the Experiment
    # Capture the time for all the experiment
    v_begin_exp = time.time()

    # Experiment Name
    experiment_name = "exp5_aleatory_linear_circle"

    # Create the 'outputs' directory if it doesn't exist
    output_folder = 'outputs'
    os.makedirs(output_folder, exist_ok=True)

    # create an experiment directory by parameters
    directory_path = (output_folder + "/" + experiment_name + "_"
                      + str(N_LOCAL_NETWORKS_MIN) + "_"
                      + str(N_LOCAL_NETWORKS_MAX)
                      + "_" + str(N_SAMPLES))
    os.makedirs(directory_path, exist_ok=True)

    # generate the experiment data file in csv
    file_path = directory_path + '/data.csv'

    # Erase the file if exists
    if os.path.exists(file_path):
        os.remove(file_path)
        print("Existing file deleted:", file_path)

    # Begin the process
    for n_local_networks in range(N_LOCAL_NETWORKS_MIN, N_LOCAL_NETWORKS_MAX + 1):  # 5
        for i_sample in range(1, N_SAMPLES + 1):  # 1 - 1000 , 1, 2
            # generate the aleatory local network template
            d_variable_cnf_function, l_var_exit = PathCircleTemplate.generate_aleatory_template(
                n_var_network=N_VAR_NETWORK)
            for v_topology in [4, 3]:
                l_data_sample = []
                print("Experiment", i_sample, "of", N_SAMPLES, " TOPOLOGY:", v_topology)

                o_cbn = PathCircleTemplate.generate_cbn_from_template(v_topology=v_topology,
                                                                      d_variable_cnf_function=d_variable_cnf_function,
                                                                      l_var_exit=l_var_exit,
                                                                      n_local_networks=n_local_networks,
                                                                      n_var_network=N_VAR_NETWORK)

                # find attractors
                v_begin_find_attractors = time.time()
                o_cbn.find_local_attractors_optimized()
                v_end_find_attractors = time.time()
                n_time_find_attractors = v_end_find_attractors - v_begin_find_attractors

                # find the compatible pairs
                v_begin_find_pairs = time.time()
                o_cbn.find_compatible_pairs()
                v_end_find_pairs = time.time()
                n_time_find_pairs = v_end_find_pairs - v_begin_find_pairs

                # Find attractor fields
                v_begin_find_fields = time.time()
                o_cbn.find_attractor_fields()
                v_end_find_fields = time.time()
                n_time_find_fields = v_end_find_fields - v_begin_find_fields

                # collect indicators
                d_collect_indicators = {
                    # initial parameters
                    "i_sample": i_sample,
                    "n_local_networks": n_local_networks,
                    "n_var_network": N_VAR_NETWORK,
                    "v_topology": v_topology,
                    "n_output_variables": N_OUTPUT_VARIABLES,
                    "n_clauses_function": N_CLAUSES_FUNCTION,
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

                # save the collected indicator to analysis
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
                pickle_path = directory_path + '/cbn_' + str(i_sample) + '_' + str(v_topology) + ".pkl"
                with open(pickle_path, 'wb') as file:
                    # Use pickle.dump to save the object to the file
                    pickle.dump(o_cbn, file)

                # Close the file
                file.close()
                print("Pickle object saved in:", pickle_path)

                CustomText.print_duplex_line()
            CustomText.print_stars()
        CustomText.print_dollars()

    # Take the time of the experiment
    v_end_exp = time.time()
    v_time_exp = v_end_exp - v_begin_exp
    print("Time experiment (in seconds): ", v_time_exp)

    print("End experiment")


run_script()
