# import libraries
from classes.cbnetwork import CBN


# pass the parameters
n_local_networks = 5
n_var_network = 5
n_relations = 2
relations_fixed = False

# create a Coupled Boolean Network with the parameters
o_cbn = CBN.generate_aleatory_cbn(n_local_networks= n_local_networks, n_var_network=n_var_network,
                                  n_relations=n_relations, relations_fixed=relations_fixed)
