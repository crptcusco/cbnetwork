# external imports
import copy
import multiprocessing

# local imports
from classes.cbnetwork import CBN
from classes.globaltopology import GlobalTopology
from classes.localtemplates import LocalNetworkTemplate

# save the number of CPUs
NUM_CPUS = multiprocessing.cpu_count()

# pass the parameters
N_LOCAL_NETWORKS = 6
N_VARS_NETWORK = 20
N_OUTPUT_VARS = 2
N_INPUT_VARS = 2
V_TOPOLOGY = 2
N_MAX_CLAUSES = 2
N_MAX_LITERALS = 2

# GENERATE THE LOCAL NETWORK TEMPLATE
o_template = LocalNetworkTemplate(n_vars_network=N_VARS_NETWORK, n_input_variables=N_INPUT_VARS,
                                  n_output_variables=N_OUTPUT_VARS, n_max_of_clauses=N_MAX_CLAUSES,
                                  n_max_of_literals=N_MAX_LITERALS, v_topology=V_TOPOLOGY)

# GENERATE THE GLOBAL TOPOLOGY
o_global_topology = GlobalTopology.generate_sample_topology(v_topology=V_TOPOLOGY,
                                                            n_nodes=N_LOCAL_NETWORKS)

# generate aleatory CBN by topology
o_cbn = CBN.generate_cbn_from_template(v_topology=V_TOPOLOGY,
                                       n_local_networks=N_LOCAL_NETWORKS,
                                       n_vars_network=N_VARS_NETWORK,
                                       o_template=o_template,
                                       l_global_edges=o_global_topology.l_edges)

# Show the Topology
o_cbn.plot_topology()

# Find the local attractors
o_cbn.find_local_attractors_parallel_with_weigths(num_cpus=NUM_CPUS)

# Find the compatible pairs
o_cbn.find_compatible_pairs_parallel_with_weights(num_cpus=NUM_CPUS)

# Mount the stable attractor fields
o_cbn.mount_stable_attractor_fields_parallel_chunks(num_cpus=NUM_CPUS)

# Show local attractors
o_cbn.show_local_attractors()

# Show the attractors pairs
o_cbn.show_attractor_pairs()

# Show the attractors fields
o_cbn.show_attractors_fields()

