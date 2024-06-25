# local imports
from classes.cbnetwork import CBN
from classes.utils.customtext import CustomText

CustomText.make_principal_title('SCRIPT TO TEST ALL THE FUNCTIONALITIES')

# pass the parameters
N_LOCAL_NETWORKS = 10
N_VAR_NETWORK = 10
N_OUTPUT_VARIABLES = 2
N_INPUT_VARIABLES = 2
V_TOPOLOGY = 3

# generate aleatory CBN by topology
o_cbn = CBN.generate_aleatory_cbn_by_topology(n_local_networks=N_LOCAL_NETWORKS, n_var_network=N_VAR_NETWORK,
                                              v_topology=V_TOPOLOGY, n_output_variables=N_OUTPUT_VARIABLES)

o_cbn.find_local_attractors_sequential()
o_cbn.find_compatible_pairs()
o_cbn.mount_stable_attractor_fields()


# Testing methods
o_cbn.show_local_attractors()
o_cbn.show_local_attractors_dictionary()
o_cbn.show_attractor_pairs()
o_cbn.show_attractors_fields()
o_cbn.show_stable_attractor_fields_detailed()
o_cbn.show_coupled_signals_kind()
o_cbn.show_resume()

# show the kind of every coupled signal
o_cbn.show_coupled_signals_kind()

# Show the number of attractor fields by global scene
o_cbn.generate_global_scenes()
o_cbn.show_global_scenes()

# Count the attractor fields by global scene
o_cbn.count_fields_by_global_scenes()
