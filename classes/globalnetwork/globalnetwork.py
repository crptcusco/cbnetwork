# external libraries
from itertools import product

# internal libraries
from classes.utils.customtext import CustomText


class GlobalNetwork:
    def __init__(self):
        self.l_variables = []
        self.d_functions = {}

    @staticmethod
    def generate_global_network(o_cbn):
        """
        Generate a global network ( Synchronous Boolean Network )
        :param o_cbn: Coupled Boolean Network Object
        :return: an object of Synchronous Boolean Network
        """

        l_variables = []
        d_functions = {}
        for o_local_network in o_cbn.l_local_networks:
            for o_variable in o_local_network.l_variables:
                # Save the variables
                l_variables.append(o_variable)
                # Save the relations
                # Future Work
        print(l_variables)

    @classmethod
    def transform_attractor_fields_to_global_states(cls, l_attractor_fields):
        # Future Work
        pass

    @staticmethod
    def test_attractor_fields(o_cbn):
        b_flag = True
        for o_attractor_field in o_cbn.l_attractor_fields:
            if o_attractor_field.test_global_dynamic():
                print("Attractor Field", o_attractor_field.l_index, ": Passed")
            else:
                print("Attractor Field", o_attractor_field.l_index, ": Failed")
        return b_flag

    @staticmethod
    def generate_global_states(o_attractor_field, o_cbn):
        global_states = []
        for attractor_index in o_attractor_field.l_attractor_indexes:
            o_local_attractor = o_cbn.get_local_attractor_by_index(attractor_index)
            for o_state in o_local_attractor.states:
                global_states.append(o_state)

        o_attractor_field.l_global_states = global_states

    def test_global_dynamic(self):
        pass
        # for global_state in self.
        #     return True
        # return False


class GlobalState:
    def __init__(self, l_values):
        self.l_values = l_values


class GlobalAttractor:
    def __init__(self,l_global_states):
        self.l_global_states = l_global_states

#
# class AttractorField:
#     def __init__(self, index, l_attractor_indexes):
#         self.index = index
#         self.l_attractor_indexes = l_attractor_indexes
#         # order the attractor indexes
#         self.l_attractor_indexes.sort(key=lambda x: x.index)
#         # calculate properties
#         self.l_global_states = None
#
#     def show(self):
#         print("Attractor Field Index: ", self.index)
#         print(self.l_attractor_indexes)


