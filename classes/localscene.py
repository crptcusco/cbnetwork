class LocalScene:
    def __init__(self, index, l_values=None, l_index_signals=None):
        """
        Initialize a LocalScene object.

        Args:
            index (int): The index of the scene.
            l_values (list, optional): List of values associated with the scene. Defaults to None.
            l_index_signals (list, optional): List of index signals associated with the scene. Defaults to None.
        """
        self.index = index
        self.l_values = l_values if l_values is not None else []
        self.l_index_signals = l_index_signals if l_index_signals is not None else []
        # List of attractors in the scene
        self.l_attractors = []


class LocalAttractor:
    def __init__(self, g_index, l_index, l_states, network_index, relation_index=None, local_scene=None):
        """
        Initialize a LocalAttractor object.

        Args:
            g_index (int): Global index of the attractor.
            l_index (int): Local index of the attractor.
            l_states (list): List of LocalState objects representing the states of the attractor.
            network_index (int): Index of the network where the attractor is located.
            relation_index (int, optional): Index of the relation associated with the attractor. Defaults to None.
            local_scene (LocalScene, optional): LocalScene object associated with the attractor. Defaults to None.
        """
        self.g_index = g_index
        self.l_index = l_index
        self.l_states = l_states
        self.network_index = network_index
        self.relation_index = relation_index
        self.local_scene = local_scene

    def show(self):
        """
        Display detailed information about the LocalAttractor object.

        Prints the network index, input signal index, local scene, global index, local index, and states.
        """
        print("Network Index:", self.network_index, ", Input Signal Index:", self.relation_index,
              ", Scene:", self.local_scene, ", Global Index:", self.g_index, ", Local Index:", self.l_index, ", States:", end="")
        for o_state in self.l_states:
            print(end='[')
            for variable in o_state.l_variable_values:
                print(variable, end=",")
            print(end=']')
        print()

    def show_short(self):
        """
        Display a brief overview of the LocalAttractor object.

        Prints the network index, attractor index, and states.
        """
        print("Net. Index:", self.network_index, ", Attrac. Index:", self.l_index, ", States:", end="")
        for o_state in self.l_states:
            print(end='[')
            for variable in o_state.l_variable_values:
                print(variable, end=",")
            print(end=']')
        print()


class LocalState:
    def __init__(self, l_variable_values):
        """
        Initialize a LocalState object.

        Args:
            l_variable_values (list): List of variable values for the state.
        """
        self.l_variable_values = l_variable_values
