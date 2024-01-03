# VARIABLE MODEL ONLY HAVE VARIABLE_NAME, CNF FUNCTION
class InternalVariable:
    index = 0
    cnf_function = []

    def __init__(self, index, cnf_function):
        self.index = int(index)
        self.cnf_function = cnf_function

    def show(self):
        print("Variable Index: " + str(self.index) + " -> CNF :" + str(self.cnf_function))

