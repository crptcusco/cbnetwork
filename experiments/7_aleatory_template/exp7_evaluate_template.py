from classes.cbnetwork import CBN
from classes.topologytemplate import TopologyTemplate

# Constants for the test
N_VAR_NETWORK = 5
N_LOCAL_NETWORKS = 6
V_TOPOLOGY = 6

# generate the aleatory local network template object
o_topology_template = TopologyTemplate.generate_aleatory_template(n_var_network=N_VAR_NETWORK, v_topology=6)
o_topology_template.show()

CBN.show_allowed_topologies()

# generate a linear CBN from the template
o_aleatory_cbn = o_topology_template.generate_cbn_from_template(v_topology=V_TOPOLOGY,
                                                                n_local_networks=N_LOCAL_NETWORKS)
