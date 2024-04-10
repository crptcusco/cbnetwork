from classes.cbnetwork import CBN

# SCRIPT TO TEST THE NEW FUNCTIONALITIES

# CONSTANTS
N_LOCAL_NETWORKS = 6
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

o_cbn.show_description()

o_cbn.find_local_attractors_optimized()
o_cbn.show_local_attractors()

# show kind of coupling signals
o_cbn.show_coupled_signals_kind()

o_cbn.show_resume()

# generate the global scenes
o_cbn.generate_global_scenes()
o_cbn.show_global_scenes()

# generate a global transition from one transition to other
# Generar un stado global
# generate a global state aleatorio o manual
o_cbn.total_variables = o_cbn.n_local_networks * o_cbn.get_n_local_variables()
o_cbn.initial_state = [0] * o_cbn.total_variables
dict_manual_global_state = []
