class LocalScene:
    def __init__(self, l_values=None):
        self.l_values = l_values
        # Calculated properties
        self.l_attractors = []


class LocalAttractor:
    def __init__(self, index, l_states):
        self.index = index
        self.l_states = l_states

    def show(self):
        print("INFO:", "Index -", self.index, "states - ", self.l_states)


class LocalState:
    def __init__(self, l_variable_values):
        self.l_variable_values = l_variable_values
