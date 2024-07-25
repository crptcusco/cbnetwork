# internal imports
from classes.cbnetwork import CBN

# external imports

# Parameters
N_LOCAL_NETWORKS = 6
N_VARS_NETWORK = 5
V_TOPOLOGY = 2
N_EDGES = 10
N_INPUT_VARS = 2
N_OUTPUT_VARS = 2
N_MAX_CLAUSES = 2
N_MAX_LITERALS = 2

o_cbn = CBN.cbn_generator(v_topology=V_TOPOLOGY, n_edges=N_EDGES, n_local_networks=N_LOCAL_NETWORKS,
                          n_input_variables=N_INPUT_VARS, n_output_variables=N_OUTPUT_VARS,
                          n_vars_network=N_VARS_NETWORK, n_max_of_clauses=N_MAX_CLAUSES,
                          n_max_of_literals=N_MAX_LITERALS)

o_cbn.show_description()
