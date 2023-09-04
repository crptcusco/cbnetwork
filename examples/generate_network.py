# import libraries

import cbn

# pass the parameters
num_of_local_networks = 5
num_of_input_relations = 2
num_relations_fixed = True

# create a Coupled Boolean Network with the parameters
o_cbn = cbn(num_of_local_networks, num_of_input_relations, num_relations_fixed)

