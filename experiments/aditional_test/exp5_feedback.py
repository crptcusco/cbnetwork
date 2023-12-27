# external imports
import random

import ray
import time
import pandas as pd
import numpy as np

# local imports
from classes.cbnetwork import CBN
from classes.internalvariable import InternalVariable
from classes.localnetwork import LocalNetwork
from classes.utils.customtext import CustomText

"""
Experiment 1 - Test the ring structure 
using aleatory generated networks
number of local networks 3 - 10 
"""


def generate_aleatory_template(n_var_network):
    # basic properties
    index = 0
    l_var_intern = list(range(n_var_network + 1, (n_var_network * 2) + 1))
    l_var_exit = random.sample(range(1, n_var_network + 1), 2)
    l_var_external = [n_var_network * 2 + 1]
    # print(l_var_intern)
    # print(l_var_saida)
    # print(l_var_external)

    # calculate properties
    l_var_total = l_var_intern + l_var_external

    # generate the aleatory dynamic
    d_variable_cnf_function = {}
    b_flag = True
    while b_flag:
        for i_variable in l_var_intern:
            # generate cnf function
            d_variable_cnf_function[i_variable] = random.sample(l_var_total, 3)
            d_variable_cnf_function[i_variable] = [-elemento if random.choice([True, False]) else elemento for elemento
                                                   in d_variable_cnf_function[i_variable]]
        # check if any function has the coupling signal
        for key, value in d_variable_cnf_function.items():
            if l_var_external[0] or -l_var_external[0] in value:
                b_flag = False

    # for key, value in d_variable_cnf_function.items():
    #     print(key, "->", value)

    return d_variable_cnf_function, l_var_exit


def generate_local_networks_dynamic_from_template(l_local_networks, l_directed_edges, n_input_variables,
                                                  o_local_network_template):
    # GENERATE THE DYNAMICS OF EACH LOCAL NETWORK
    number_max_of_clauses = 2
    number_max_of_literals = 3

    # generate an auxiliary list to modify the variables
    l_local_networks_updated = []

    # update the dynamic for every local network
    for o_local_network in l_local_networks:
        # Create a list of all RDDAs variables
        l_aux_variables = []
        # find the directed edges by network index
        l_input_signals = CBN.find_input_edges_by_network_index(o_local_network.index, l_directed_edges)
        # add the variable index of the directed edges
        for o_signal in l_input_signals:
            l_aux_variables.append(o_signal.index_variable)
        # add local variables
        l_aux_variables.extend(o_local_network.l_var_intern)

        # generate the function description of the variables
        des_funct_variables = []
        # generate clauses
        for i_local_variable in o_local_network.l_var_intern:
            l_clauses_node = []
            for v_clause in range(0, random.randint(1, number_max_of_clauses)):
                v_num_variable = random.randint(1, number_max_of_literals)
                # randomly select from the signal variables
                l_literals_variables = random.sample(l_aux_variables, v_num_variable)
                l_clauses_node.append(l_literals_variables)
            # adding the description of variable in object form
            o_variable_model = InternalVariable(i_local_variable, l_clauses_node)
            # adding the description in functions of every variable
            des_funct_variables.append(o_variable_model)
        # adding the local network to a list of local networks
        o_local_network.des_funct_variables = des_funct_variables.copy()
        l_local_networks_updated.append(o_local_network)
        print("Local network created :", o_local_network.index)
        CustomText.print_simple_line()

        # actualized the list of local networks
        return l_local_networks_updated


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
N_DIRECTED_EDGES = 1

# verbose parameters
SHOW_MESSAGES = True

# generate an special CBN
l_local_networks = []
l_directed_edges = []

# generate the aleatory local network template
d_variable_cnf_function, l_var_exit = generate_aleatory_template(n_var_network=N_VAR_NETWORK)

print(l_var_exit)
for key, value in d_variable_cnf_function.items():
    print(key, "->", value)

# # Generate the local_networks
# N_LOCAL_NETWORKS = 5
# l_local_networks, l_directed_edges = generate_local_networks_dynamic_from_template(l_local_networks=l_local_networks,
#                                                                                    l_directed_edges=l_directed_edges,
#                                                                                    n_input_variables=N_INPUT_VARIABLES,
#                                                                                    o_local_network_template=
#                                                                                    o_local_network_template)
#
# # generate the special coupled boolean network
# o_special_cbn = CBN(l_local_networks=l_local_networks,
#                     l_directed_edges=l_directed_edges)

# @staticmethod
#     def generate_aleatory_template(n_var_network, n_input_variables, n_directed_edges):
#         # basic properties
#         index = 0
#         l_var_intern = list(range(1,n_input_variables+1))
#         l_var_extern = random.sample(list(range(-n_var_network-1,-1)), 2)
#
#         # lista of number of clauses
#         l_clauses = [1, 2]
#
#         d_variable_cnf_function = {}
#         for i_variable in l_var_intern:
#             # generate aleatory dynamic
#             # number of clauses
#
#         # variables functions in CNF format
#         d_variable_cnf_function = {1: [[2, 3], [1, -15]],
#                                    2: [[1, 15]],
#                                    3: [[3, -1, 15]],
#                                    4: [[-5, 6, 7]],
#                                    5: [[6, -7, -16]],
#
#
#         # generating the local network dynamic
#         l_var_intern.process_input_signals(l_input_signals)
#         for i_local_variable in l_var_intern.l_var_intern:
#             o_variable_model = InternalVariable(i_local_variable, d_variable_cnf_function[i_local_variable])
#             l_var_intern.des_funct_variables.append(o_variable_model)
#
#         # Processed properties
#         self.des_funct_variables = []
#         self.l_var_exterm = []
#         self.l_var_total = []
#         self.num_var_total = 0
#         self.dic_var_cnf = {}
#
#         self.l_input_signals = []
#         self.l_output_signals = []
#
#         # Calculated properties
#         self.l_local_scenes = []
#
#
#
#         d_template = {}
#         return d_template
#     # TAREA PARA LA CUSCO!!!!


# o_local_network_template.show()

# o_cbn = CBN.generate_cbn(N_LOCAL_NETWORKS=N_LOCAL_NETWORKS, n_var_network=N_VAR_NETWORK, v_topology=V_TOPOLOGY,
#                          n_output_variables=N_OUTPUT_VARIABLES, n_input_variables=N_INPUT_VARIABLES,
#                          o_local_network_template=o_local_network_template)
#
#
# o_cbn = CBN.generate_cbn(,,,,)
#
# o_cbn = generate_aleatory_cbn_linear(
#     N_LOCAL_NETWORKS=10,
#     n_var_network=5,
#     v_equal_local_networks=True,
#     n_output_variables=2,
#     n_input_variables=3,
#     n_clauses_function=1)


# o_cbn.show_cbn_graph()

# # Begin the Experiment
# # Capture the time for all the experiment
# v_begin_exp = time.time()
#
# # Begin the process
# l_data_sample = []
#
# for N_LOCAL_NETWORKS in range(N_LOCAL_NETWORKS_MIN, n_local_networks_max):
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
#             o_cbn = CBN.generate_cbn(N_LOCAL_NETWORKS=N_LOCAL_NETWORKS,
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
#                 "N_LOCAL_NETWORKS": N_LOCAL_NETWORKS,
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
