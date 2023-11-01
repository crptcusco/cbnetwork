# import libraries
from classes.cbnetwork import CBN

print("MESSAGE:", "CBN ALEATORY SCRIPT EXAMPLE")
print("=======================================")

# pass the parameters
n_local_networks = 6
n_var_network = 5
n_output_variables = 2
n_clauses_function = 2
v_topology = 6

# create a Coupled Boolean Network with the parameters
o_cbn = CBN.generate_cbn(n_local_networks=n_local_networks, n_var_network=n_var_network, v_topology=v_topology,
                         n_output_variables=n_output_variables, n_clauses_function=n_clauses_function)

# find local network attractors
o_cbn.find_attractors()

# find the compatible pairs
o_cbn.find_compatible_pairs()

# show the kind of every coupled signal
o_cbn.show_coupled_signals_kind()

# find attractors fields
# o_cbn.find_attractor_fields()

# # show graph with networkx
# o_cbn.generate_graph()

# # show attractors
# o_cbn.show_attractors()

# Show attractor fields
# o_cbn.show_attractors_fields()

