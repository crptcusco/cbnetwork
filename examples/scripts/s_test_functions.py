from classes.cbnetwork import CBN

# SCRIPT TO TEST THE NEW FUNCTIONALITIES

# CONSTANTS
N_LOCAL_NETWORKS = 5
N_LOCAL_VARIABLES = 5
V_TOPOLOGY = 4
N_OUTPUT_VARIABLES = 2
N_INPUT_VARIABLES = 2

# GENERATE THE NETWORK OBJECT
o_cbn = CBN.generate_aleatory_cbn_by_topology(n_local_networks=N_LOCAL_NETWORKS,
                                              n_var_network=N_LOCAL_VARIABLES,
                                              v_topology=V_TOPOLOGY,
                                              n_output_variables=N_OUTPUT_VARIABLES,
                                              n_input_variables=N_INPUT_VARIABLES)

o_cbn.show_cbn()

o_cbn.find_local_attractors_optimized()
o_cbn.show_local_attractors()

# o_cbn.plot_global_graph()

# CBN.show_allowed_topologies()
