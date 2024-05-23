from classes.cbnetwork import CBN
from classes.topologytemplate import AleatoryTemplate

# Constants for the test
N_VAR_NETWORK = 5
N_LOCAL_NETWORKS = 6

# generate the aleatory local network template object
o_aleatory_template = AleatoryTemplate.generate_aleatory_template(n_var_network=N_VAR_NETWORK)
o_aleatory_template.show()

CBN.show_allowed_topologies()

# generate a linear CBN from the template
o_aleatory_cbn = o_aleatory_template.generate_cbn_from_template(v_topology=5, n_local_networks=N_LOCAL_NETWORKS)

