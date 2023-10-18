# import libraries
from classes.cbnetwork import CBN
from classes.directededge import DirectedEdge
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
    l_index_local_networks.append(o_local_network)

# generate the directed edges
o_directed_edge1 = DirectedEdge(2,
                                1,
                                [4, 5],
                                15,
                                " 4 v 5 ")

o_directed_edge2 = DirectedEdge(3,
                                2,
                                [8, 9],
                                16,
                                " 4 v 5 ")

o_directed_edge3 = DirectedEdge(2,
                                1,
                                [4, 5],
                                17,
                                " 4 v 5 ")

o_directed_edge4 = DirectedEdge(2,
                                1,
                                [4, 5],
                                18,
                                " 4 v 5 ")

# for t_edge in l_edges:
#     coupling_function = " " + " ∨ ".join(list(map(str, l_output_variables))) + " "
#     o_directed_edge = DirectedEdge(o_local_network.index,
#                                    o_local_network_co.index,
#                                    l_output_variables,
#                                    v_cont_var,
#                                    coupling_function)
#     # generate the object of the directed edge
#     o_directed_edge = DirectedEdge(t_edge[0], t_edge[1], )

# generating the CBN network
o_cbn = CBN(l_local_networks, l_directed_edges)

# # Find attractors
# o_cbn.find_attractors()
#
# # generate the global scenes
# o_cbn.generate_global_scenes()
# # o_cbn.show_global_scenes()
#
# # find the compatible pairs
# o_cbn.find_compatible_pairs()

# # show graph with networkx
# o_cbn.generate_graph()

# # show attractors
# o_cbn.show_attractors()

# Find attractors fields
# o_cbn.find_attractor_fields()

# Show attractor fields
# o_cbn.show_attractors_fields()

print("MESSAGE:", "END SCRIPT EXAMPLE")
