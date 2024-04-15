from classes.cbnetwork import CBN

# Script to test the Global Network class

# Parameters
N_LOCAL_NETWORKS = 6
N_VAR_NETWORK = 5
N_INPUT_VARIABLES = 2
N_OUTPUT_VARIABLES = 2
V_TOPOLOGY = 3

# Generate a Random CBN object
o_cbn = CBN.generate_aleatory_cbn_by_topology(n_local_networks=N_LOCAL_NETWORKS,
                                              n_var_network=N_VAR_NETWORK,
                                              n_input_variables=N_INPUT_VARIABLES,
                                              n_output_variables=N_OUTPUT_VARIABLES,
                                              v_topology=V_TOPOLOGY)
o_cbn.show_description()

# Find the global stable states
o_cbn.find_local_attractors_sequential()
o_cbn.find_compatible_pairs()
o_cbn.find_stable_attractor_fields()
# o_cbn.show_stable_attractor_fields()

# Generate Global Network

# transform the local attractor fields to global stable states
l_global_stable_states = []
for o_attractor_field in o_cbn.l_attractor_fields:
    stable_state_index_attractors = []
    for t_pair in o_attractor_field:
        stable_state_index_attractors.append(t_pair[0].index)
        stable_state_index_attractors.append(t_pair[1].index)
    print(stable_state_index_attractors)
