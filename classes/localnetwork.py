class LocalNetwork:
    def __init__(self, num_local_network, l_var_intern, l_input_signals=None, description_variables=None):
        if l_input_signals is None:
            l_input_signals = []
        if description_variables is None:
            description_variables = []

        self.index = num_local_network
        self.l_var_intern = l_var_intern

        self.l_input_signals = l_input_signals
        self.description_variables = description_variables

    def show(self):
        print('Local Network', self.index)
        print('Variables intern : ', self.l_var_intern)
        for o_signal in self.l_input_signals:
            o_signal.show()
        pass
        # Description variables

        for o_variable in self.description_variables:
            o_variable.show()

    def process_parameters(self):
        pass

