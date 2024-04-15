# local imports
from classes.cbnetwork import CBN
from classes.utils.customtext import CustomText

# script to put a manual parameters for the example of 4 networks
print("LINEAL CBN ALEATORY SCRIPT EXAMPLE")
CustomText.print_duplex_line()

# pass the parameters
N_LOCAL_NETWORKS = 6
N_VAR_NETWORK = 5
N_OUTPUT_VARIABLES = 2
N_INPUT_VARIABLES = 2
V_TOPOLOGY = 3

o_cbn = CBN.generate_aleatory_cbn_by_topology(n_local_networks=N_LOCAL_NETWORKS,
                                              n_var_network=N_VAR_NETWORK,
                                              v_topology=V_TOPOLOGY,
                                              n_output_variables=N_OUTPUT_VARIABLES)

o_cbn.find_local_attractors_heap()
# o_cbn.show_attractors()

o_cbn.find_compatible_pairs()
# o_cbn.show_attractor_pairs()

o_cbn.mount_stable_attractor_fields()
# o_cbn.show_attractors_fields()

# show kind of coupling signals
o_cbn.show_coupled_signals_kind()
