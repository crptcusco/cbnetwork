# import libraries
from classes.cbnetwork import CBN

print("CBN script example")
print("List of the Allowed Topologies")

# allowed_topologies = {
#     1: "complete_graph",
#     2: "binomial_tree",
#     3: "cycle_graph",
#     4: "path_graph"
# }

print(CBN.show_allowed_topologies())

# pass the parameters
n_local_networks = 5
n_var_network = 4
# n_relations = 2
# relations_fixed = True
n_output_variables = 2
n_clauses_function = 2
v_topology = 1

# create a Coupled Boolean Network with the parameters
o_cbn = CBN.generate_cbn(n_local_networks=n_local_networks, n_var_network=n_var_network, v_topology=v_topology,
                         n_output_variables=n_output_variables, n_clauses_function=n_clauses_function)

o_cbn.generate_graph()

# Find attractors
o_cbn.find_attractors()

# show attractors
o_cbn.show_attractors()

# Find attractors fields
# o_cbn.find_attractor_fields()

# Show attractor fields
# o_cbn.show_attractors_fields()

print("END")

