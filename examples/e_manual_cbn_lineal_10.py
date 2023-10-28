# import libraries
from classes.cbnetwork import CBN
from classes.directededge import DirectedEdge
from classes.internalvariable import InternalVariable
from classes.localnetwork import LocalNetwork

# script to put a manual parameters for the example of 4 networks
print("MESSAGE:", "LINEAL CBN MANUAL SCRIPT EXAMPLE")
print("==============================+++++++")

# pass the parameters
l_local_networks = []
l_directed_edges = []

n_local_networks = 10
n_variables = 5
# l_index_local_networks = list(range(1, n_local_networks))


# generate the variables 5 per network in sequence
d_network_variables = {i: list(range(n_variables * (i - 1) + 1, n_variables * i + 1)) for i in range(1, 11)}

# generate the edges of the linear CBN
l_edges = [(1, 2), (2, 3), (3, 4), (5, 6), (7, 8), (9, 10)]

# generate the networks
for i_local_net in d_network_variables.keys():
    # generate the Local network
    o_local_network = LocalNetwork(i_local_net, d_network_variables[i_local_net])
    l_local_networks.append(o_local_network)
    # Show the networks
    o_local_network.show()

# generate the directed edges
cont_output_variable = 0
index_variable_signal = (n_local_networks * n_variables) + 1
for t_edge in l_edges:
    l_output_variables = [4, 5]
    # generate coupling function
    coupling_function = " " + " ∨ ".join(map(str, l_output_variables)) + " "
    print(coupling_function)
    o_directed_edge = DirectedEdge(t_edge[0],
                                   t_edge[1],
                                   [x + cont_output_variable for x in l_output_variables],
                                   index_variable_signal,
                                   coupling_function)
    print(o_directed_edge.l_output_variables)
    l_directed_edges.append(o_directed_edge)
    cont_output_variable += 5
    index_variable_signal += 1
    # o_directed_edge.show()

d_variable_cnf_function = {i: [[x for x in range(i*5+1, i*5+6)],
                               [x for x in range((i-1)*5+1, (i-1)*5+6, 2)]] for i in range(1, 51)}

for key, value in d_variable_cnf_function.items():
    print(key, "->", value)

# d_variable_cnf_function = {1: [[2, 3], [1, -15]],
#                            2: [[1, 15]],
#                            3: [[3, -1, 15]],
#                            4: [[-5, 6, 7]],
#                            5: [[6, -7, -16]],
#                            6: [[-4, -5, 16]],
#                            7: [[-5, 16, 7]],
#                            8: [[9, 10, 17]],
#                            9: [[8, 18]],
#                            10: [[8, 9]],
#                            11: [[-12, 13]],
#                            12: [[11, 13]],
#                            13: [[14, -11, 12]],
#                            14: [[11, -12]]}

# # generating the local network dynamic
# for o_local_network in l_local_networks:
#     l_input_signals = DirectedEdge.find_input_edges_by_network_index(o_local_network.index, l_directed_edges)
#     o_local_network.process_input_signals(l_input_signals)
#     for i_local_variable in o_local_network.l_var_intern:
#         o_variable_model = InternalVariable(i_local_variable, d_variable_cnf_function[i_local_variable])
#         o_local_network.des_funct_variables.append(o_variable_model)
#
#
# print(l_local_networks)
#
# # generating the CBN network
# o_cbn = CBN(l_local_networks, l_directed_edges)
#
# # Find attractors
# o_cbn.find_attractors()
#
# # show attractors
# o_cbn.show_attractors()
#
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
