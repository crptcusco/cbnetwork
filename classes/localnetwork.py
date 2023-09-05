class LocalNetwork:
    def __init__(self, num_local_network, l_var_intern, l_input_relations=None, l_intern_dynamic=None):
        if l_input_relations is None:
            l_input_relations = []
            self.l_signals = l_input_relations
        self.index = num_local_network
        self.l_var_intern = l_var_intern

    def show(self):
        print('Local Network', self.index)
        print('Variables intern : ', self.l_var_intern)
        for o_signal in self.l_signals:
            o_signal.show()
        pass