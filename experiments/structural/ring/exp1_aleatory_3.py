# external imports
import ray
import time
import pandas as pd
import numpy as np

# local imports
from classes.cbnetwork import CBN
from classes.utils.customtext import CustomText

"""
Experiment 1 - Test the ring structure 
using aleatory generated networks
number of local networks 3  
"""

# experiment parameters
n_samples = 3
n_local_networks_min = 3
n_local_networks_max = 10
n_var_network = 5
n_output_variables = 2
v_topology = 3  # cycle graph
n_clauses_function = 2

# Ray Configurations
# ray.shutdown()
# runtime_env = {"working_dir": "/home/reynaldo/Documents/RESEARCH/SynEstRDDA", "pip": ["requests", "pendulum==2.1.2"]}
# ray.init(address='ray://172.17.163.253:10001', runtime_env=runtime_env, log_to_driver=False)
# ray.init(address='ray://172.17.163.244:10001', runtime_env=runtime_env , log_to_driver=False, num_cpus=12)
# ray.init(log_to_driver=False, num_cpus=12)

# Begin Experiment

# Capture the time for all the experiment
v_begin_exp = time.time()


# Begin the process


l_data_sample = []
for n_local_networks in range(n_local_networks_min, n_local_networks_max):
    for i_sample in range(1, n_samples + 1):
        print("Experiment", i_sample, "of", n_samples)
        # generate a Coupled Boolean Network with the parameters
        o_cbn = CBN.generate_cbn(n_local_networks=n_local_networks,
                                 n_var_network=n_var_network,
                                 v_topology=v_topology,
                                 n_output_variables=n_output_variables,
                                 n_clauses_function=n_clauses_function)
        # Find attractors
        o_cbn.find_local_attractors_optimized_method()
        # find the compatible pairs
        o_cbn.find_compatible_pairs()
        # Find attractor fields
        o_cbn.find_attractor_fields()

        # calculate indicator
        # list of number of attractors by network
        # list of pairs attractor by coupling signal
        # list of attractor fields

        # collect indicators
        d_collect_indicators = {
            # initial parameters
            "i_sample": i_sample,
            "n_local_networks": n_local_networks,
            "n_var_network": n_var_network,
            "v_topology": v_topology,
            "n_output_variables": n_output_variables,
            "n_clauses_function": n_clauses_function,
            # calculate parameters
            "n_local_attractors": o_cbn.get_n_local_attractors(),
            "n_pair_attractors": o_cbn.get_n_pair_attractors(),
            "n_attractor_fields": o_cbn.get_n_attractor_fields()
        }
        l_data_sample.append(d_collect_indicators)
        # show the important outputs
        # o_cbn.show_resume()
CustomText.print_duplex_line()
# Take the time of the experiment
v_end_exp = time.time()
v_time_exp = v_end_exp - v_begin_exp
print("Time experiment (in seconds): ", v_time_exp)

# Save the collected indicator to analysis
pf_res = pd.DataFrame(l_data_sample)
pf_res.reset_index(drop=True, inplace=True)

# Save the experiment data in csv, using pandas Dataframe
path = "exp1_aleatory_3.csv"
pf_res.to_csv(path)
print("Experiment saved in:", path)

print("End experiment")
