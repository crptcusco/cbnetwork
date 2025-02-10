# external imports
import random

# local imports
from classes.cnflist import CNFList


class LocalNetworkTemplate:
    def __init__(self, n_vars_network, n_input_variables, n_output_variables,
                 n_max_of_clauses=None, n_max_of_literals=None, v_topology=1):
        """
        Initialize a LocalNetworkTemplate object.

        Args:
            v_topology: Topology of the network.
            n_vars_network (int): Number of variables in the network.
            n_input_variables (int): Number of input variables.
            n_output_variables (int): Number of output variables.
            n_max_of_clauses (int, optional): Maximum number of clauses for CNF functions. Defaults to None.
            n_max_of_literals (int, optional): Maximum number of literals for CNF functions. Defaults to None.
        """
        # Fixed Parameters
        self.v_topology = v_topology
        self.n_vars_network = n_vars_network
        self.n_input_variables = n_input_variables
        self.n_output_variables = n_output_variables
        self.n_max_of_clauses = n_max_of_clauses
        self.n_max_of_literals = n_max_of_literals

        # Calculated Parameters
        self.l_output_var_indexes = []
        self.d_variable_cnf_function = {}
        self.generate_local_dynamic()

    def generate_local_dynamic(self):
        """
        Generate CNF functions and output variable indexes dynamically.
        """
        # Internal variables indices
        l_internal_var_indexes = list(range(self.n_vars_network + 1, (self.n_vars_network * 2) + 1))

        # Indices for input coupling signals
        l_input_coupling_signal_indexes = [self.n_vars_network * 2 + 1]

        # Generate CNF function for each internal variable
        l_input_variables = random.sample(l_internal_var_indexes, self.n_input_variables)

        for i_variable in l_internal_var_indexes:
            input_coup_sig_index = None
            if i_variable in l_input_variables:
                input_coup_sig_index = random.choice(l_input_coupling_signal_indexes)

            # Generate CNF function for the variable
            self.d_variable_cnf_function[i_variable] = CNFList.generate_cnf(
                l_inter_vars=l_internal_var_indexes,
                input_coup_sig_index=input_coup_sig_index,
                max_clauses=self.n_max_of_clauses,
                max_literals=self.n_max_of_literals
            )

        # Generate output variable indexes
        self.l_output_var_indexes = random.sample(range(1, self.n_vars_network + 1), self.n_output_variables)

    def show(self):
        """
        Display information about the LocalNetworkTemplate.
        """
        print("Local Network Template")
        print("-" * 50)
        print("Local dynamic:")
        for key, value in self.d_variable_cnf_function.items():
            print(key, ":", value)
        print("Output variables for coupling signal:", self.l_output_var_indexes)

    def get_output_variables_from_template(self, i_local_network, l_local_networks):
        """
        Retrieve output variables from the template based on the local network index.

        Args:
            i_local_network (int): Index of the local network.
            l_local_networks (list): List of local network objects.

        Returns:
            list: List of output variables for the specified local network.
        """
        l_variables = []
        for o_local_network in l_local_networks:
            if o_local_network.index == i_local_network:
                for position in self.l_output_var_indexes:
                    l_variables.append(o_local_network.l_var_intern[position - 1])

        return l_variables
