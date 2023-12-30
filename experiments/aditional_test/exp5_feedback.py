# external imports
import random

import ray
import time
import pandas as pd
import numpy as np

# local imports
from classes.cbnetwork import CBN
from classes.directededge import DirectedEdge
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


def get_output_variables_from_template(i_local_network, l_local_networks, l_var_exit):
    # select the internal variables
    l_variables = []
    for o_local_network in l_local_networks:
        if o_local_network.index == i_local_network:
            # select the specific variables from variable list intern
            for position in l_var_exit:
                l_variables.append(o_local_network.l_var_intern[position - 1])

    return l_variables


def generate_local_dynamic_with_template(l_local_networks, l_directed_edges, d_variable_cnf_function):
    # GENERATE THE DYNAMICS OF EACH LOCAL NETWORK
    number_max_of_clauses = 2
    number_max_of_literals = 3
    n_local_networks = len(l_local_networks)
    n_local_variables = len(l_local_networks[0].l_var_intern)

    # generate an auxiliary list to modify the variables
    l_local_networks_updated = []
    print(l_local_networks)
    # update the dynamic for every local network
    for o_local_network in l_local_networks:
        print("Local Network:", o_local_network.index)
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
            for key, value in d_variable_cnf_function.items():
                print(key, "->", value)
            # adapt the template function to the specific network
            print(i_local_variable)
            print(i_local_variable - ((o_local_network.index-1) * n_local_variables))
            l_clauses_node = d_variable_cnf_function[i_local_variable - ((o_local_network.index-1) * n_local_variables)+5]

            # ACTUALIZAR LOS VALORES DE LAS CLAUSULAS TAMBIEN!!!
            print(l_clauses_node)
            # TAREA CUSCO!!!

            # generate an internal variable from satispy
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

# generate a special CBN
l_directed_edges = []
#############################################

# generate the aleatory local network template
d_variable_cnf_function, l_var_exit = generate_aleatory_template(n_var_network=N_VAR_NETWORK)

n_local_networks = 10
# generate the local networks with the indexes and variables (without relations or dynamics)
l_local_networks = CBN.generate_local_networks_indexes_variables(n_local_networks, N_VAR_NETWORK)

# generate the CBN topology
l_relations = CBN.generate_cbn_topology(n_local_networks, V_TOPOLOGY)

# search the last variable from the local network variables
i_last_variable = l_local_networks[-1].l_var_intern[-1]

# generate the directed edges given the last variable generated and the selected output variables
for relation in l_relations:
    # get the output variables from template
    l_output_variables = get_output_variables_from_template(relation[0], l_local_networks, l_var_exit)

    # generate the coupling function
    coupling_function = " " + " ∨ ".join(list(map(str, l_output_variables))) + " "
    # generate the Directed-Edge object
    o_directed_edge = DirectedEdge(index_variable_signal=i_last_variable,
                                   input_local_network=relation[1],
                                   output_local_network=relation[0],
                                   l_output_variables=l_output_variables,
                                   coupling_function=coupling_function)
    i_last_variable += 1
    l_directed_edges.append(o_directed_edge)

# Process the coupling signals for every local network
for o_local_network in l_local_networks:
    # find the signals for every local network
    l_input_signals = CBN.find_input_edges_by_network_index(o_local_network.index, l_directed_edges)
    o_local_network.process_input_signals(l_input_signals)

# generate dynamic of the local networks with template
l_local_networks = generate_local_dynamic_with_template(l_local_networks, l_directed_edges, d_variable_cnf_function)

# # generate the special coupled boolean network
o_special_cbn = CBN(l_local_networks=l_local_networks, l_directed_edges=l_directed_edges)
# o_special_cbn.show_cbn()


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
