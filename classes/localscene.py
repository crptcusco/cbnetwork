class LocalScene:
    def __init__(self, l_values=None, l_index_signals=None):
        self.l_values = l_values
        self.l_index_signals = l_index_signals
        # Calculated properties
        self.l_attractors = []


class LocalAttractor:
    def __init__(self, index, l_states, network_index, relation_index=None, local_scene=None):
        self.index = index
        self.l_states = l_states

        self.network_index = network_index
        self.relation_index = relation_index
        self.local_scene = local_scene

    def show(self):
        print("INFO:", "Network Index -", self.network_index, ", Input Signal Index -", self.relation_index,
              ", Scene -", self.local_scene, ", Attractor Index -", self.index, ", States - ", end="")
        for o_state in self.l_states:
            print(end='[')
            for variable in o_state.l_variable_values:
                print(variable, end=",")
            print(end=']')
        print()


class LocalState:
    def __init__(self, l_variable_values):
        self.l_variable_values = l_variable_values
