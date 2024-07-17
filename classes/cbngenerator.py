# external imports
import random

# local imports
from classes.cbnetwork import CBN
from classes.directededge import DirectedEdge
from classes.globaltopology import GlobalTopology
from classes.internalvariable import InternalVariable
from classes.utils.customtext import CustomText


class CBNGenerator:
    def __init__(self, v_topology, o_local_network_template):
        self.v_topology = v_topology
        self.o_local_network_template = o_local_network_template


class LocalNetworkTemplate:
    def __init__(self, l_variables, l_input_variables, l_output_variables):
        self.l_variables = l_variables
        self.l_input_variables = l_input_variables
        self.l_output_variables = l_output_variables

    def show(self):
        CustomText.make_title('Local Template description')
        print("Variables List:", self.l_variables)
        print('Input Variables List:', self.l_input_variables)
        print('Output Variables List:', self.l_output_variables)

    @staticmethod
    def generate_template(n_variables, n_input_variables, n_output_variables):
        # generate the list of variables
        l_variables = list(range(1, n_variables+1))
        # generate the list of input variables
        l_input_variables = random.sample(l_variables, n_input_variables)
        # generate the list of output variables
        l_output_variables = random.sample(l_variables, n_output_variables)

        o_local_network_template = LocalNetworkTemplate(l_variables, l_input_variables, l_output_variables)
        return o_local_network_template


# parameters
N_VARIABLES = 5
N_INPUT_VARIABLES = 2
N_OUTPUT_VARIABLES = 2
V_TOPOLOGY = 2

# o_local_network_template = LocalNetworkTemplate
o_local_template = LocalNetworkTemplate.generate_template(n_variables=N_VARIABLES,
                                                          n_input_variables=N_INPUT_VARIABLES,
                                                          n_output_variables=N_OUTPUT_VARIABLES)
o_local_template.show()

# select the topology
GlobalTopology.show_allowed_topologies()
# generate the global edges
l_global_edges = GlobalTopology.generate_edges(v_topology=V_TOPOLOGY, n_nodes=10, n_edges=10, o_graph=None, seed=None)















