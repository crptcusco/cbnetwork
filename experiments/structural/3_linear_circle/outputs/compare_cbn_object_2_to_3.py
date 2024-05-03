import pickle
from classes.cbnetwork import CBN
from classes.directededge import DirectedEdge
from classes.localnetwork import LocalNetwork
from classes.localscene import LocalScene

# Abre el archivo pickle en modo lectura binaria ('rb')
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_1_3.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_1_4.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_2_3.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_2_4.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_3_3.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_3_4.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_4_3.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_4_4.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_5_3.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_5_4.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_6_3.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_6_4.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_7_3.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_7_4.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_8_3.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_8_4.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_9_3.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_9_4.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_10_3.pkl'
# path_data = '2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_10_4.pkl'

path_data = ""
count_test = 1
b_all_test = True
# Generate and iterate over the list of file paths
for i in range(1, 11):
    for j in range(3, 5):
        # path_data = f'2_0_data_slow/exp5_aleatory_linear_circle_8_8_10/pkl_cbn/cbn_{i}_{j}.pkl'
        path_data = f'2_0_data_slow/exp5_aleatory_linear_circle_3_3_10/pkl_cbn/cbn_{i}_{j}.pkl'
        print(path_data)

        with open(path_data, 'rb') as f:
            # load the data from the pickle file
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

        # test attractors
        n_cbn_attractors = 0
        for o_local_network in o_cbn.l_local_networks:
            for o_scene in o_local_network.l_local_scenes:
                for o_attractor in o_scene.l_attractors:
                    n_cbn_attractors += 1

        b_attractors = False
        if n_cbn_attractors == len(aux_cbn.d_local_attractors.items()):
            b_attractors = True

        print("Number Local Attractors:", n_cbn_attractors)
        print("Number Local Attractors:", len(aux_cbn.d_local_attractors.items()))
        print('Test Passed:', b_attractors)

        # Test for stable attractor pairs
        b_attractor_pairs = False

        n_pairs = 0
        for o_directed_edge in o_cbn.l_directed_edges:
            n_pairs += len(o_directed_edge.d_comp_pairs_attractors_by_value[0])
            n_pairs += len(o_directed_edge.d_comp_pairs_attractors_by_value[1])

        aux_n_pairs = aux_cbn.get_n_attractor_pairs()

        if aux_n_pairs == n_pairs:
            b_attractor_pairs = True

        print("Number Stable Attractor Pairs:", aux_n_pairs)
        print("Number Stable Attractor Pairs:", n_pairs)
        print('Test Passed:', b_attractor_pairs)

        # attractor fields
        b_attractor_fields = False
        if len(aux_cbn.l_attractor_fields) == len(o_cbn.l_attractor_fields):
            b_attractor_fields = True

        print("Number Stable Attractor Fields:", len(aux_cbn.l_attractor_fields))
        print("Number Stable Attractor Fields:", len(o_cbn.l_attractor_fields))
        print('Test Passed:', b_attractor_fields)

        b_all_test = b_all_test and b_attractors and b_attractor_pairs and b_attractor_fields
        if b_all_test:
            print(f"TEST {count_test} PASSED")
        else:
            print(f"TEST {count_test} FAILED")
        count_test += 1
        print('*'*80)

if b_all_test:
    print('ALL TESTS PASSED')
else:
    print('ALL TESTS FAILED')
