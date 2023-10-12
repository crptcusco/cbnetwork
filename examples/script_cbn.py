# import libraries
from classes.cbnetwork import CBN

print("MESSAGE:", "CBN script example")
print("MESSAGE:", "List of the Allowed Topologies")

# allowed_topologies = {
#     1: "complete_graph",
#     2: "binomial_tree",
#     3: "cycle_graph",
#     4: "path_graph"
# }

CBN.show_allowed_topologies()

# pass the parameters
n_local_networks = 6
n_var_network = 5
# n_relations = 2
# relations_fixed = True
n_output_variables = 2
n_clauses_function = 2
v_topology = 6

# create a Coupled Boolean Network with the parameters
o_cbn = CBN.generate_cbn(n_local_networks=n_local_networks, n_var_network=n_var_network, v_topology=v_topology,
                         n_output_variables=n_output_variables, n_clauses_function=n_clauses_function)

# Find attractors
o_cbn.find_attractors()

# generate the global scenes
o_cbn.generate_global_scenes()
# o_cbn.show_global_scenes()

# find the compatible pairs
o_cbn.find_compatible_pairs()

# # show graph with networkx
# o_cbn.generate_graph()

# # show attractors
# o_cbn.show_attractors()

# Find attractors fields
# o_cbn.find_attractor_fields()

# Show attractor fields
# o_cbn.show_attractors_fields()

