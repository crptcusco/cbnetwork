# VARIABLE MODEL ONLY HAVE VARIABLE_NAME, CNF FUNCTION
class InternalVariable:
    index = 0
    # list_interacts = []
    cnf_function = []

    def __init__(self, index, cnf_function):
        self.index = int(index)
        self.cnf_function = cnf_function

        print('Internal Variable', self.index, 'created')

    def show(self):
        print("Variable Index: " + str(self.index) + " -> CNF :" + str(self.cnf_function))

        # print("====================================")
        # print("Show the function for every variable")
        # for key, value in d_var_cnf_func.items():
        #     print(key, "->", value)
