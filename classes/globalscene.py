class GlobalScene:
    def __init__(self, index, l_signal_indexes, l_values_signals):
        self.index = index
        self.l_signal_indexes = l_signal_indexes
        self.l_values_signals = l_values_signals

    def show(self):
        print("-------------------------------")
        print("Index Global Scene:", self.index)
        print("Indexes Directed Edges:", self.l_signal_indexes)
        print("Directed Edges Values:", self.l_values_signals)
