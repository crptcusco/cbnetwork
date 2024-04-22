from classes.cbnetwork import CBN
from classes.utils.customtext import CustomText

CustomText.make_principal_title('SCRIPT TO TEST ALL THE FUNCTIONALITIES')

# CONSTANTS
N_LOCAL_NETWORKS = 6
N_LOCAL_VARIABLES = 5
V_TOPOLOGY = 3
N_OUTPUT_VARIABLES = 2
N_INPUT_VARIABLES = 2

# GENERATE THE NETWORK OBJECT
o_cbn = CBN.generate_aleatory_cbn_by_topology(n_local_networks=N_LOCAL_NETWORKS,
                                              n_var_network=N_LOCAL_VARIABLES,
                                              n_input_variables=N_INPUT_VARIABLES,
                                              n_output_variables=N_OUTPUT_VARIABLES,
                                              v_topology=V_TOPOLOGY)

# Show CBN Description
o_cbn.show_description()

# Find Local Attractors
o_cbn.find_local_attractors_sequential()
# o_cbn.show_local_attractors()

o_cbn.generate_attractor_dictionary()
# o_cbn.show_attractors_dictionary()

# # Find compatible attractor pairs
# o_cbn.find_compatible_pairs()
# o_cbn.show_attractor_pairs()
#
# # Mount stable attractor fields
# o_cbn.mount_stable_attractor_fields()
# o_cbn.show_stable_attractor_fields()
#
# # show kind of coupling signals
# o_cbn.show_coupled_signals_kind()
#
# # Show resume
# o_cbn.show_resume()



