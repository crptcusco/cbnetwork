# import libraries
import random

from classes.cbnetwork import CBN
from classes.directededge import DirectedEdge
from classes.internalvariable import InternalVariable
from classes.localnetwork import LocalNetwork

# script to put a manual parameters for the example of 4 networks
print("MESSAGE:", "LINEAL CBN ALEATORY SCRIPT EXAMPLE")
print("==============================+++++++")

# pass the parameters
n_local_networks = 10
n_var_network = 5
n_output_variables = 2
n_clauses_function = 1
v_topology = 7

# create a Coupled Boolean Network with the parameters
o_cbn = CBN.generate_cbn(n_local_networks=n_local_networks, n_var_network=n_var_network, v_topology=v_topology,
                         n_output_variables=n_output_variables, n_clauses_function=n_clauses_function)

# Find attractors
o_cbn.find_attractors()

# show the kind of the edges
o_cbn.show_directed_edges()

# # show attractors
# o_cbn.show_attractors()

# # # generate the global scenes
# # o_cbn.generate_global_scenes()
#
# # # Show global attractors
# # o_cbn.show_global_scenes()
#
# # find the compatible pairs
# o_cbn.find_compatible_pairs()
#
# # show attractor pairs
# o_cbn.show_attractor_pairs()
#
# # # show graph with networkx
# # o_cbn.generate_graph()
#
# # # show attractors
# # o_cbn.show_attractors()
#
# # Find attractors fields
# # o_cbn.find_attractor_fields()
#
# # Show attractor fields
# # o_cbn.show_attractors_fields()
#
# print("==============================")
# print("MESSAGE:", "END SCRIPT EXAMPLE")
