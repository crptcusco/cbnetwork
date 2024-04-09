# import internal
from classes.cbnetwork import CBN
from classes.directededge import DirectedEdge
from classes.internalvariable import InternalVariable
from classes.localnetwork import LocalNetwork
from classes.utils.customtext import CustomText

# external imports
import parsl
parsl.load()

# script to put a manual parameters for the example of 4 networks
print("MESSAGE:", "TEST PARALLEL - LINEAL CBN MANUAL SCRIPT EXAMPLE")
CustomText.print_duplex_line()

# pass the parameters
l_local_networks = []
l_directed_edges = []

n_local_nets = 10
n_var_net = 5
n_total_var = n_local_nets * n_var_net

# generate the 5 variables per network in sequence
d_network_variables = {i: list(range(n_var_net * (i - 1) + 1, n_var_net * i + 1)) for i in range(1, n_local_nets + 1)}

# generate the edges of the 1_linear CBN
l_edges = [(i, i + 1) for i in range(1, 10)]

# generate the networks
for i_local_net in d_network_variables.keys():
    # generate the Local network
    o_local_network = LocalNetwork(i_local_net, d_network_variables[i_local_net])
    l_local_networks.append(o_local_network)
    # Show the local network
    o_local_network.show()

# generate the directed edges
cont_output_variable = 0
index_variable_signal = (n_local_nets * n_var_net) + 1
for t_edge in l_edges:
    l_output_variables = [4 + cont_output_variable, 5 + cont_output_variable]
    # generate coupling function
    coupling_function = " " + " ∧ ".join(map(str, l_output_variables)) + " "
    # generate the Directed Edge object
    o_directed_edge = DirectedEdge(index_variable_signal, t_edge[1], t_edge[0], l_output_variables, coupling_function)
    # add the directed object to list
    l_directed_edges.append(o_directed_edge)
    # updating the count of variables
    cont_output_variable += 5
    # updating the index variable signal
    index_variable_signal += 1

# Generate the functions for every variable in the CBN
d_var_cnf_func = {}
count_network = 1
count_var = 0
for o_local_network in l_local_networks:
    d_var_cnf_func[count_var + 1] = [[count_var + 2, -(count_var + 3), count_var + 4]]
    d_var_cnf_func[count_var + 2] = [[count_var + 1, -(count_var + 3), -(count_var + 5)]]
    d_var_cnf_func[count_var + 3] = [[-(count_var + 2), -(count_var + 4), count_var + 5]]
    if o_local_network.index == 1:
        d_var_cnf_func[count_var + 4] = [[count_var + 3, count_var + 5]]
        d_var_cnf_func[count_var + 5] = [[count_var + 1, count_var + 2]]
    else:
        d_var_cnf_func[count_var + 4] = [[count_var + 3, count_var + 5, n_total_var + o_local_network.index - 1]]
        d_var_cnf_func[count_var + 5] = [[-(count_var + 1), count_var + 2, n_total_var + o_local_network.index - 1]]
    count_var += 5
    count_network += 1

# show the function for every variable
for key, value in d_var_cnf_func.items():
    print(key, "->", value)

# generating the local network dynamic
for o_local_network in l_local_networks:
    l_input_signals = CBN.find_input_edges_by_network_index(o_local_network.index, l_directed_edges)
    o_local_network.process_input_signals(l_input_signals)
    for i_local_variable in o_local_network.l_var_intern:
        o_variable_model = InternalVariable(i_local_variable, d_var_cnf_func[i_local_variable])
        o_local_network.des_funct_variables.append(o_variable_model)

# generating the CBN network
o_cbn = CBN(l_local_networks, l_directed_edges)

# Show the CBN Information
o_cbn.show_description()

# Find local attractors parallelized with Parsl
tasks1 = CBN.find_local_attractors_parsl(o_cbn.l_local_networks)
# Wait for all tasks to complete
o_cbn.l_local_networks = [task.result() for task in tasks1]

# Process
for o_local_network in o_cbn.l_local_networks:
    o_cbn.process_local_attractors(o_local_network)

# Show local attractors after all tasks have completed
o_cbn.show_local_attractors()

# Find attractor pairs parallelized with Parsl
tasks2 = CBN.find_compatible_pairs_parsl(o_cbn)
# Wait for all tasks to complete
o_cbn.l_directed_edges = [task.result() for task in tasks2]
o_cbn.show_attractor_pairs()

# Find and show stable attractor fields
print("normal function")
# o_cbn.find_attractor_fields()
print("parallel function")
o_cbn.find_stable_attractor_fields_parsl()
o_cbn.show_stable_attractor_fields()
