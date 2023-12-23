# external imports
import random

import ray
import time
import pandas as pd
import numpy as np

# local imports
from classes.cbnetwork import CBN
from classes.localnetwork import LocalNetwork
from classes.utils.customtext import CustomText

"""
Experiment 1 - Test the ring structure 
using aleatory generated networks
number of local networks 3 - 10 
"""

# Ray Configurations
# ray.shutdown()
# runtime_env = {"working_dir": "/home/reynaldo/Documents/RESEARCH/SynEstRDDA", "pip": ["requests", "pendulum==2.1.2"]}
# ray.init(address='ray://172.17.163.253:10001', runtime_env=runtime_env, log_to_driver=False)
# ray.init(address='ray://172.17.163.244:10001', runtime_env=runtime_env , log_to_driver=False, num_cpus=12)
# ray.init(log_to_driver=False, num_cpus=12)

# experiment parameters
N_SAMPLES = 100
N_LOCAL_NETWORKS_MIN = 3
N_LOCAL_NETWORKS_MAX = 10
N_VAR_NETWORK = 5
N_OUTPUT_VARIABLES = 2
N_INPUT_VARIABLES = 2
V_TOPOLOGY = 3  # cycle graph
N_CLAUSES_FUNCTION = 2

# verbose parameters
SHOW_MESSAGES = True

# Generate the local network template
n_local_networks = 5
# generate the template for the local network
o_local_network_template = LocalNetwork.generate_template(N_VAR_NETWORK)

o_cbn = CBN.generate_cbn(n_local_networks=n_local_networks, n_var_network=N_VAR_NETWORK, v_topology=V_TOPOLOGY,
                         n_output_variables=N_OUTPUT_VARIABLES, n_input_variables=N_INPUT_VARIABLES,
                         o_local_network_template=o_local_network_template)


o_cbn = CBN.generate_cbn(,,,,)

o_cbn = generate_aleatory_cbn_linear(
    n_local_networks=10,
    n_var_network=5,
    v_equal_local_networks=True,
    n_output_variables=2,
    n_input_variables=3,
    n_clauses_function=1)


o_cbn.show_cbn_graph()

# # Begin the Experiment
# # Capture the time for all the experiment
# v_begin_exp = time.time()
#
# # Begin the process
# l_data_sample = []
#
# for n_local_networks in range(N_LOCAL_NETWORKS_MIN, n_local_networks_max):
#     for i_sample in range(1, N_SAMPLES + 1):
#         # generate the local network components
#         d_des_local_network = generate_aleatory_local_network(N_VAR_NETWORK, N_OUTPUT_VARIABLES, N_INPUT_VARIABLES)
#         for V_TOPOLOGY in ["linear", "ring"]:
#             print("Experiment", i_sample, "of", N_SAMPLES)
#
#             # generate the coupled boolean network for the specific topology
#             if V_TOPOLOGY == "linear":
#                 t_cbn = generate_aleatory_cbn_linear(d_des_local_network)
#                 o_cbn = CBN(t_cbn[0], t_cbn[1])
#             else:
#                 t_cbn = generate_aleatory_cbn_ring(d_des_local_network)
#                 o_cbn = CBN(t_cbn[0], t_cbn[1])
#
#             # generate a Coupled Boolean Network with the parameters
#             o_cbn = CBN.generate_cbn(n_local_networks=n_local_networks,
#                                      N_VAR_NETWORK=N_VAR_NETWORK,
#                                      V_TOPOLOGY=V_TOPOLOGY,
#                                      N_OUTPUT_VARIABLES=N_OUTPUT_VARIABLES,
#                                      N_CLAUSES_FUNCTION=N_CLAUSES_FUNCTION)
#
#             # Find attractors
#             v_begin_find_attractors = time.time()
#             o_cbn.find_local_attractors_optimized()
#             v_end_find_attractors = time.time()
#             n_time_find_attractors = v_end_find_attractors - v_begin_find_attractors
#
#             # find the compatible pairs
#             v_begin_find_pairs = time.time()
#             o_cbn.find_compatible_pairs()
#             v_end_find_pairs = time.time()
#             n_time_find_pairs = v_end_find_pairs - v_begin_find_pairs
#
#             # Find attractor fields
#             v_begin_find_fields = time.time()
#             o_cbn.find_attractor_fields()
#             v_end_find_fields = time.time()
#             n_time_find_fields = v_end_find_fields - v_begin_find_fields
#
#             # collect indicators
#             d_collect_indicators = {
#                 # initial parameters
#                 "i_sample": i_sample,
#                 "n_local_networks": n_local_networks,
#                 "N_VAR_NETWORK": N_VAR_NETWORK,
#                 "V_TOPOLOGY": V_TOPOLOGY,
#                 "N_OUTPUT_VARIABLES": N_OUTPUT_VARIABLES,
#                 "N_CLAUSES_FUNCTION": N_CLAUSES_FUNCTION,
#                 # calculate parameters
#                 "n_local_attractors": o_cbn.get_n_local_attractors(),
#                 "n_pair_attractors": o_cbn.get_n_pair_attractors(),
#                 "n_attractor_fields": o_cbn.get_n_attractor_fields(),
#                 # time parameters
#                 "n_time_find_attractors": n_time_find_attractors,
#                 "n_time_find_pairs": n_time_find_pairs,
#                 "n_time_find_fields": n_time_find_fields
#             }
#             l_data_sample.append(d_collect_indicators)
#             # show the important outputs
#             # o_cbn.show_resume()
# CustomText.print_duplex_line()
# # Take the time of the experiment
# v_end_exp = time.time()
# v_time_exp = v_end_exp - v_begin_exp
# print("Time experiment (in seconds): ", v_time_exp)
#
# # Save the collected indicator to analysis
# pf_res = pd.DataFrame(l_data_sample)
# pf_res.reset_index(drop=True, inplace=True)
#
# # Save the experiment data in csv, using pandas Dataframe
# path = "exp1_aleatory.csv"
# pf_res.to_csv(path)
# print("Experiment saved in:", path)
#
# print("End experiment")