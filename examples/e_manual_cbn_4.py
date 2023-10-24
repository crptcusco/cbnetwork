# import libraries
from classes.cbnetwork import CBN
from classes.directededge import DirectedEdge
from classes.internalvariable import InternalVariable
from classes.localnetwork import LocalNetwork

# script to put a manual parameters for the example of 4 networks
print("MESSAGE:", "CBN MANUAL SCRIPT EXAMPLE")
print("==============================+++++++")

# pass the parameters
l_local_networks = []
l_directed_edges = []
l_index_local_networks = list(range(1, 5))
d_network_variables = {1: [1, 2, 3],
                       2: [4, 5, 6, 7],
                       3: [8, 9, 10],
                       4: [11, 12, 13, 14]}

l_edges = [(2, 1), (3, 2), (2, 3), (4, 3)]

# generate the networks
for i_local_net in d_network_variables.keys():
    # print("Network:", i_local_net)
    # for i_variable in d_network_variables[i_local_net]:
    #     print("Variable", i_variable)
    # generate the Local network
    o_local_network = LocalNetwork(i_local_net, d_network_variables[i_local_net])
    l_local_networks.append(o_local_network)

# generate the directed edges
o_directed_edge1 = DirectedEdge(1,
                                2,
                                [4, 5],
                                15,
                                " 4 ∨ 5 ")

o_directed_edge2 = DirectedEdge(2,
                                3,
                                [8, 9],
                                16,
                                " 8 ∨ 9 ")

o_directed_edge3 = DirectedEdge(3,
                                2,
                                [6, 7],
                                17,
                                " 6 ∨ 7 ")

o_directed_edge4 = DirectedEdge(3,
                                4,
                                [13, 14],
                                18,
                                " 13 ∨ 14 ")

l_directed_edges.append(o_directed_edge1)
l_directed_edges.append(o_directed_edge2)
l_directed_edges.append(o_directed_edge3)
l_directed_edges.append(o_directed_edge4)


d_variable_cnf_function = {1: [[2, 3], [1, -15]],
                           2: [[1, 15]],
                           3: [[3, -1, 15]],
                           4: [[-5, 6, 7]],
                           5: [[6, -7, -16]],
                           6: [[-4, -5, 16]],
                           7: [[-5, 16, 7]],
                           8: [[9, 10, 17]],
                           9: [[8, 18]],
                           10: [[8, 9]],
                           11: [[-12, 13]],
                           12: [[11, 13]],
                           13: [[14, -11, 12]],
                           14: [[11, -12]]}

# generating the local network dynamic
for o_local_network in l_local_networks:
    l_input_signals = DirectedEdge.find_input_edges_by_network_index(o_local_network.index, l_directed_edges)
    o_local_network.process_input_signals(l_input_signals)
    for i_local_variable in o_local_network.l_var_intern:
        o_variable_model = InternalVariable(i_local_variable, d_variable_cnf_function[i_local_variable])
        o_local_network.des_funct_variables.append(o_variable_model)


print(l_local_networks)

# generating the CBN network
o_cbn = CBN(l_local_networks, l_directed_edges)

# Find attractors
o_cbn.find_attractors()

# show attractors
o_cbn.show_attractors()

# # generate the global scenes
# o_cbn.generate_global_scenes()

# # Show global attractors
# o_cbn.show_global_scenes()

# find the compatible pairs
o_cbn.find_compatible_pairs()

# show attractor pairs
# o_cbn.show_attractor_pairs()

# # show graph with networkx
# o_cbn.generate_graph()

# # show attractors
# o_cbn.show_attractors()

# Find attractors fields
# o_cbn.find_attractor_fields()

# Show attractor fields
# o_cbn.show_attractors_fields()

print("==============================")
print("MESSAGE:", "END SCRIPT EXAMPLE")
