class GlobalScene:
    def __init__(self, l_indexes, values):
        """
        Initialize a GlobalScene object.

        Args:
            l_indexes (list): List of indexes.
            values (list): List of values.
        """
        self.l_indexes = l_indexes
        self.values = values
        self.number_attractor_fields = 0

    def show(self):
        """
        Display the details of the GlobalScene object.

        Prints the list of indexes and values stored in the object.
        """
        print(self.l_indexes)
        print(self.values)

