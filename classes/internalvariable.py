# VARIABLE MODEL ONLY HAVE VARIABLE_NAME, CNF FUNCTION
class InternalVariable:
    variable_name = 0
    # list_interacts = []
    cnf_function = []

    def __init__(self, name_variable, cnf_function):
        self.variable_name = int(name_variable)
        self.cnf_function = cnf_function

    def show(self):
        print("V: " + str(self.variable_name) + " -> CNF :" + str(self.cnf_function))

        # print("====================================")
        # print("Show the function for every variable")
        # for key, value in d_var_cnf_func.items():
        #     print(key, "->", value)
