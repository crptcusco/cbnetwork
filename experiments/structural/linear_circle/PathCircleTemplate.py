# external imports
import random
import ray
import time
import pandas as pd
import numpy as np

# local imports
from classes.cbnetwork import CBN
from classes.directededge import DirectedEdge
from classes.internalvariable import InternalVariable
from classes.localnetwork import LocalNetwork
from classes.utils.customtext import CustomText


class PathCircleTemplate:
    def __init__(self):
        pass

    @staticmethod
    def generate_aleatory_template(n_var_network):
        """
        Generates aleatory template for a local network
        :param n_var_network:
        :return: Dictionary of cnf function for variable and list of exit variables
        """

        # basic properties
        index = 0
        l_var_intern = list(range(n_var_network + 1, (n_var_network * 2) + 1))
        l_var_exit = random.sample(range(1, n_var_network + 1), 2)
        l_var_external = [n_var_network * 2 + 1]

        # calculate properties
        l_var_total = l_var_intern + l_var_external

        # generate the aleatory dynamic
        d_variable_cnf_function = {}
        b_flag = True
        while b_flag:
            for i_variable in l_var_intern:
                # generate cnf function
                d_variable_cnf_function[i_variable] = random.sample(l_var_total, 3)
                d_variable_cnf_function[i_variable] = [
                    [-element if random.choice([True, False]) else element for element
                     in d_variable_cnf_function[i_variable]]]
            # check if any function has the coupling signal
            for key, value in d_variable_cnf_function.items():
                if l_var_external[0] or -l_var_external[0] in value:
                    b_flag = False

        return d_variable_cnf_function, l_var_exit

    @staticmethod
    def get_output_variables_from_template(i_local_network, l_local_networks, l_var_exit):
        # select the internal variables
        l_variables = []
        for o_local_network in l_local_networks:
            if o_local_network.index == i_local_network:
                # select the specific variables from variable list intern
                for position in l_var_exit:
                    l_variables.append(o_local_network.l_var_intern[position - 1])

        return l_variables

    @staticmethod
    def update_clause_from_template(l_local_networks, o_local_network, i_local_variable, d_variable_cnf_function,
                                    l_directed_edges, v_topology):
        """
        update clause from template
        :param v_topology:
        :param l_local_networks:
        :param o_local_network:
        :param i_local_variable:
        :param d_variable_cnf_function:
        :return: l_clauses_node
        """

        l_indexes_directed_edges = []
        for o_directed_edge in l_directed_edges:
            l_indexes_directed_edges.append(o_directed_edge.index_variable)

        # find the correct cnf function for the variables
        n_local_variables = len(l_local_networks[0].l_var_intern)
        i_template_variable = i_local_variable - ((o_local_network.index - 1) * n_local_variables) + n_local_variables
        pre_l_clauses_node = d_variable_cnf_function[i_template_variable]

        print("Local Variable index:", i_local_variable)
        print("Template Variable index:", i_template_variable)
        print("CNF Function:", pre_l_clauses_node)

        # for every pre-clause update the variables of the cnf function
        l_clauses_node = []
        for pre_clause in pre_l_clauses_node:
            # update the number of the variable
            l_clause = []
            for template_value in pre_clause:
                # evaluate if the topology is linear(4) and is the first local network and not in the list of dictionary
                if (v_topology == 4 and o_local_network.index == 1
                        and abs(template_value) not in list(d_variable_cnf_function.keys())):
                    continue
                else:
                    # save the symbol (+ or -) of the value True for "+" and False for "-"
                    b_symbol = True
                    if template_value < 0:
                        b_symbol = False
                    # replace the value with the variable index
                    local_value = abs(template_value) + (
                            (o_local_network.index - 3) * n_local_variables) + n_local_variables
                    # analyzed if the value is an external value,searching the value in the list of intern variables
                    if local_value not in o_local_network.l_var_intern:
                        # print(o_local_network.l_var_intern)
                        # print(o_local_network.l_var_exterm)
                        # print(local_value)
                        local_value = o_local_network.l_var_exterm[0]
                    # add the symbol to the value
                    if not b_symbol:
                        local_value = -local_value
                    # add the value to the local clause
                    l_clause.append(local_value)

            # add the clause to the list of clauses
            l_clauses_node.append(l_clause)

        print(i_local_variable, ":", l_clauses_node)
        return l_clauses_node

    @staticmethod
    def generate_local_dynamic_with_template(l_local_networks, l_directed_edges, d_variable_cnf_function, v_topology):
        """
        GENERATE THE DYNAMICS OF EACH LOCAL NETWORK
        :param v_topology:
        :param l_local_networks:
        :param l_directed_edges:
        :param d_variable_cnf_function:
        :return: l_local_networks updated
        """
        number_max_of_clauses = 2
        number_max_of_literals = 3

        # generate an auxiliary list to modify the variables
        l_local_networks_updated = []

        # update the dynamic for every local network
        for o_local_network in l_local_networks:
            CustomText.print_simple_line()
            print("Local Network:", o_local_network.index)

            # find the directed edges by network index
            l_input_signals_by_network = CBN.find_input_edges_by_network_index(index=o_local_network.index,
                                                                               l_directed_edges=l_directed_edges)

            # # add the variable index of the directed edges
            # for o_signal in l_input_signals_by_network:
            #     o_local_network.l_var_exterm.append(o_signal.index_variable)
            # o_local_network.l_var_total = o_local_network.l_var_intern + o_local_network.l_var_exterm

            # generate the function description of the variables
            des_funct_variables = []
            # generate clauses for every local network adapting the template
            for i_local_variable in o_local_network.l_var_intern:
                CustomText.print_simple_line()
                # adapting the clause template to the specific variable
                l_clauses_node = PathCircleTemplate.update_clause_from_template(l_local_networks=l_local_networks,
                                                                                o_local_network=o_local_network,
                                                                                i_local_variable=i_local_variable,
                                                                                d_variable_cnf_function=d_variable_cnf_function,
                                                                                l_directed_edges=l_directed_edges,
                                                                                v_topology=v_topology)
                # generate an internal variable from satispy
                o_variable_model = InternalVariable(index=i_local_variable,
                                                    cnf_function=l_clauses_node)
                # adding the description in functions of every variable
                des_funct_variables.append(o_variable_model)

            # adding the local network to a list of local networks
            o_local_network.des_funct_variables = des_funct_variables.copy()
            l_local_networks_updated.append(o_local_network)
            print("Local network created :", o_local_network.index)
            CustomText.print_simple_line()

        # actualized the list of local networks
        return l_local_networks_updated

    @staticmethod
    def get_last_variable(l_local_networks):
        """
        search the last variable from the local network variables
        :param l_local_networks:
        :return:
        """
        last_index_variable = l_local_networks[-1].l_var_intern[-1]
        return last_index_variable

    @staticmethod
    def generate_cbn_from_template(v_topology, d_variable_cnf_function, l_var_exit, n_local_networks,n_var_network):
        """
        Generate a special CBN

        Args:
            v_topology:
            d_variable_cnf_function:
            l_var_exit:
            n_local_networks:
        Returns:
            A CBN generated from a template
        """

        # generate the local networks with the indexes and variables (without relations or dynamics)
        l_local_networks = CBN.generate_local_networks_indexes_variables(n_local_networks=n_local_networks,
                                                                         n_var_network=n_var_network)

        # generate the directed edges between the local networks
        l_directed_edges = []

        # generate the CBN topology
        l_relations = CBN.generate_cbn_topology(n_nodes=n_local_networks,
                                                v_topology=v_topology)

        # Get the last index of the variables for the indexes of the directed edges
        i_last_variable = PathCircleTemplate.get_last_variable(l_local_networks=l_local_networks) + 1

        # generate the directed edges given the last variable generated and the selected output variables
        for relation in l_relations:
            output_local_network = relation[0]
            input_local_network = relation[1]

            # get the output variables from template
            l_output_variables = PathCircleTemplate.get_output_variables_from_template(output_local_network, l_local_networks, l_var_exit)

            # generate the coupling function
            coupling_function = " " + " ∨ ".join(list(map(str, l_output_variables))) + " "
            # generate the Directed-Edge object
            o_directed_edge = DirectedEdge(index_variable_signal=i_last_variable,
                                           input_local_network=input_local_network,
                                           output_local_network=output_local_network,
                                           l_output_variables=l_output_variables,
                                           coupling_function=coupling_function)
            i_last_variable += 1
            # add the directed-edge object to the list
            l_directed_edges.append(o_directed_edge)

        # Process the coupling signals for every local network
        for o_local_network in l_local_networks:
            # find the signals for every local network
            l_input_signals = CBN.find_input_edges_by_network_index(index=o_local_network.index,
                                                                    l_directed_edges=l_directed_edges)
            # process the input signals of the local network
            o_local_network.process_input_signals(l_input_signals=l_input_signals)

        # generate dynamic of the local networks with template
        l_local_networks = PathCircleTemplate.generate_local_dynamic_with_template(l_local_networks=l_local_networks,
                                                                l_directed_edges=l_directed_edges,
                                                                d_variable_cnf_function=d_variable_cnf_function,
                                                                v_topology=v_topology)

        # generate the special coupled boolean network
        o_special_cbn = CBN(l_local_networks=l_local_networks,
                            l_directed_edges=l_directed_edges)

        return o_special_cbn
