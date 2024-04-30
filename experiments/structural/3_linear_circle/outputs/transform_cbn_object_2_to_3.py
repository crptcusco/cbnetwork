import pickle
from classes.cbnetwork import CBN
from classes.directededge import DirectedEdge
from classes.localnetwork import LocalNetwork

# Abre el archivo pickle en modo lectura binaria ('rb')
with open('2_0_data_slow/exp5_aleatory_linear_circle_3_3_10/pkl_cbn/cbn_10_3.pkl', 'rb') as f:
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
    aux_edge.show()

l_local_networks=[]
# Fix the local networks
for o_local_network in l_local_networks:
    aux_local_network = LocalNetwork(num_local_network=o_local_network.num_local_network,
                                     l_var_intern=o_local_network.l_var_intern)

    aux_local_network.des_funct_variables = o_local_network.des_funct_variables
    aux_local_network.l_var_extern = o_local_network.l_var_extern

    # # basic properties
    # self.index = num_local_network
    # self.l_var_intern = l_var_intern
    #
    # # Processed properties
    # self.des_funct_variables = []
    # self.l_var_exterm = []
    # self.l_var_total = []
    # self.num_var_total = 0
    # self.dic_var_cnf = {}
    #
    # self.l_input_signals = []
    # self.l_output_signals = []
    #
    # # Calculated properties
    # self.count_attractor = 1
    # self.l_local_scenes = []


# aux_cbn = CBN(l_local_networks=l_local_networks,l_edges)
