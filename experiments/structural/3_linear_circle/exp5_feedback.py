# external imports
import time
import pandas as pd

# local imports
from PathCircleTemplate import PathCircleTemplate
from classes.utils.customtext import CustomText

"""
Experiment 5 - Test the path and 2_ring structures 
using aleatory generated template for the local network
number of local networks 3 - 10 
"""

# experiment parameters
N_SAMPLES = 5
N_LOCAL_NETWORKS_MIN = 3
N_LOCAL_NETWORKS_MAX = 15
N_VAR_NETWORK = 5
N_OUTPUT_VARIABLES = 2
N_INPUT_VARIABLES = 2
V_TOPOLOGY = 3  # cycle graph
N_CLAUSES_FUNCTION = 2
N_DIRECTED_EDGES = 1

# verbose parameters
SHOW_MESSAGES = True

# Begin the Experiment
# Capture the time for all the experiment
v_begin_exp = time.time()

# Begin the process
l_data_sample = []

for n_local_networks in range(N_LOCAL_NETWORKS_MIN, N_LOCAL_NETWORKS_MAX+1):
    for i_sample in range(1, N_SAMPLES + 1):
        # generate the aleatory local network template
        d_variable_cnf_function, l_var_exit = PathCircleTemplate.generate_aleatory_template(n_var_network=N_VAR_NETWORK)
        for V_TOPOLOGY in [4, 3]:
            print("Experiment", i_sample, "of", N_SAMPLES, " TOPOLOGY:", V_TOPOLOGY)

            o_cbn = PathCircleTemplate.generate_cbn_from_template(v_topology=V_TOPOLOGY,
                                                                  d_variable_cnf_function=d_variable_cnf_function,
                                                                  l_var_exit=l_var_exit,
                                                                  n_local_networks=n_local_networks,
                                                                  n_var_network=N_VAR_NETWORK)

            # Find attractors
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
                "v_topology": V_TOPOLOGY,
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
            # show the important outputs

CustomText.print_duplex_line()
# Take the time of the experiment
v_end_exp = time.time()
v_time_exp = v_end_exp - v_begin_exp
print("Time experiment (in seconds): ", v_time_exp)

# Save the collected indicator to analysis
pf_res = pd.DataFrame(l_data_sample)
pf_res.reset_index(drop=True, inplace=True)

# Save the experiment data in csv, using pandas Dataframe
path = "exp5_aleatory_linear_circle.csv"
pf_res.to_csv(path)
print("Experiment saved in:", path)

print("End experiment")
