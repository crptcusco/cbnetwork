class GlobalScene:
    def __init__(self, l_indexes, values):
        self.l_indexes = l_indexes
        self.values = values
        self.number_attractor_fields = 0

    def show(self):
        print(self.l_indexes)
        print(self.values)
