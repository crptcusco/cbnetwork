import pickle
from classes.cbnetwork import CBN
from classes.directededge import DirectedEdge
from classes.localnetwork import LocalNetwork
from classes.localscene import LocalScene

# Abre el archivo pickle en modo lectura binaria ('rb')
with open('2_0_data_slow/exp5_aleatory_linear_circle_3_3_10/pkl_cbn/cbn_8_3.pkl', 'rb') as f:
    # Carga los datos del archivo pickle
    o_cbn = pickle.load(f)

# Fix the directed edge objects
l_edges = []
count_edges = 1
for o_edge in o_cbn.l_directed_edges:
    aux_edge = DirectedEdge(index=count_edges,
                            index_variable_signal=o_edge.index_variable,
                            l_output_variables=o_edge.l_output_variables,
                            input_local_network=o_edge.input_local_network,
                            output_local_network=o_edge.output_local_network,
                            coupling_function=o_edge.coupling_function)
    l_edges.append(aux_edge)
    aux_edge.show()

# Fix the local networks
l_local_networks = []
for o_local_network in o_cbn.l_local_networks:
    aux_local_network = LocalNetwork(num_local_network=o_local_network.index,
                                     l_var_intern=o_local_network.l_var_intern)

    aux_local_network.des_funct_variables = o_local_network.des_funct_variables
    aux_local_network.l_var_exterm = o_local_network.l_var_exterm
    aux_local_network.l_var_total = o_local_network.l_var_total
    aux_local_network.dic_var_cnf = o_local_network.dic_var_cnf
    aux_local_network.l_input_signals = o_local_network.l_input_signals
    aux_local_network.l_output_signals = o_local_network.l_output_signals
    aux_local_network.num_var_total = o_local_network.num_var_total

    # Fix the scenes
    # aux_local_network.l_local_scenes = o_local_network.l_local_scenes
    l_scenes = []
    for o_scene in o_local_network.l_local_scenes:
        aux_scene = LocalScene(index=o_scene.index,
                               l_values=o_scene.l_values,
                               l_index_signals=o_scene.l_index_signals)
        l_scenes.append(aux_scene)
    aux_local_network.l_local_scenes = l_scenes

    # add the local network to the
    l_local_networks.append(aux_local_network)

aux_cbn = CBN(l_local_networks=l_local_networks, l_directed_edges=l_edges)

aux_cbn.find_local_attractors_sequential()
aux_cbn.find_compatible_pairs()
aux_cbn.mount_stable_attractor_fields()

print('?'*80)
# aux_cbn.show_stable_attractor_fields()
# aux_cbn.show_stable_attractor_fields_detailed()
print("Number Stable Attractor Fields:", len(aux_cbn.l_attractor_fields))
print('?'*80)
# o_cbn.show_stable_attractor_fields()
n_pairs = 0
for o_directed_edge in o_cbn.l_directed_edges:
    n_pairs += len(o_directed_edge.d_comp_pairs_attractors_by_value[0])
    n_pairs += len(o_directed_edge.d_comp_pairs_attractors_by_value[1])
print("Number Stable Attractor Pairs:", n_pairs)
print("Number Stable Attractor Fields:", len(o_cbn.l_attractor_fields))
for o_attractor_field in o_cbn.l_attractor_fields:
    for t_pair_attractor in o_attractor_field:
        print('-'*80)
        print("Network Index:", t_pair_attractor[0].network_index,
              ", Input Signal Index:", t_pair_attractor[0].relation_index,
              ", Scene:", t_pair_attractor[0].local_scene,
              ", Local Index:", t_pair_attractor[0].index,
              ", States:", end="")
        for o_state in t_pair_attractor[0].l_states:
            print(end='[')
            for variable in o_state.l_variable_values:
                print(variable, end=",")
            print(end=']')
        print()

        print("Network Index:", t_pair_attractor[1].network_index,
              ", Input Signal Index:", t_pair_attractor[1].relation_index,
              ", Scene:", t_pair_attractor[0].local_scene,
              ", Local Index:", t_pair_attractor[1].index,
              ", States:", end="")
        for o_state in t_pair_attractor[1].l_states:
            print(end='[')
            for variable in o_state.l_variable_values:
                print(variable, end=",")
            print(end=']')
        print()
print('?'*80)
