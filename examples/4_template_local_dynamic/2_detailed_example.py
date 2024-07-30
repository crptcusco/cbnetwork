# internal imports
from classes.cbnetwork import CBN
from classes.globaltopology import GlobalTopology
from classes.localtemplates import LocalNetworkTemplate

# external imports

# Parameters
N_LOCAL_NETWORKS = 6
N_VARS_NETWORK = 5
V_TOPOLOGY = 4
N_EDGES = 10
N_INPUT_VARS = 2
N_OUTPUT_VARS = 2
N_MAX_CLAUSES = 2
N_MAX_LITERALS = 2

# GENERATE THE GLOBAL TOPOLOGY
o_global_topology = GlobalTopology.generate_sample_topology(v_topology=V_TOPOLOGY,
                                                            n_nodes=N_LOCAL_NETWORKS,
                                                            n_edges=N_EDGES)

# GENERATE THE LOCAL NETWORK TEMPLATE
o_template = LocalNetworkTemplate(v_topology=V_TOPOLOGY,
                                  n_vars_network=N_VARS_NETWORK,
                                  n_input_variables=N_INPUT_VARS,
                                  n_output_variables=N_OUTPUT_VARS,
                                  n_max_of_clauses=N_MAX_CLAUSES,
                                  n_max_of_literals=N_MAX_LITERALS)

# GENERATE THE CBN WITH THE TOPOLOGY AND TEMPLATE
o_cbn = CBN.generate_cbn_from_template(v_topology=V_TOPOLOGY,
                                       n_local_networks=N_LOCAL_NETWORKS,
                                       n_vars_network=N_VARS_NETWORK,
                                       o_template=o_template,
                                       l_global_edges=o_global_topology.l_edges)

o_cbn.show_description()

