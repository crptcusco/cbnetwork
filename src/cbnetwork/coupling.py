"""
This module defines the coupling strategies for Coupled Boolean Networks (CBNs).

It provides a flexible way to define the logical relationship between local networks
through the `CouplingStrategy` abstract base class and its concrete implementations.
"""
from abc import ABC, abstractmethod
from typing import List
from itertools import combinations

class CouplingStrategy(ABC):
    """
    Abstract base class for defining coupling strategies in a CBN.

    A coupling strategy defines the logical function that combines the output
    variables from one local network to create a single coupling signal that
    is used as an input to another local network.
    """

    @abstractmethod
    def generate_coupling_function(self, output_variables: List[int]) -> str:
        """
        Generates a string representation of the coupling function.

        For example, for an OR coupling, this might return "var1 ∨ var2".

        Args:
            output_variables (List[int]): A list of the output variable indices.

        Returns:
            str: A string representing the logical function.
        """
        pass

    @abstractmethod
    def to_cnf(self, output_variables: List[int], coupling_variable: int) -> List[List[int]]:
        """
        Converts the coupling logic into Conjunctive Normal Form (CNF).

        The CNF representation is necessary for the SAT solver to find attractors.
        This method should return a list of clauses, where each clause is a list of literals.
        For example, (A ∨ ¬B) ∧ C would be represented as [[A, -B], [C]].

        Args:
            output_variables (List[int]): A list of the output variable indices.
            coupling_variable (int): The index of the variable representing the coupling signal.

        Returns:
            List[List[int]]: A list of clauses representing the CNF.
        """
        pass

class OrCoupling(CouplingStrategy):
    """
    Represents a disjunctive (OR) coupling between local networks.

    The coupling signal is active if at least one of the output variables is active.
    """

    def generate_coupling_function(self, output_variables: List[int]) -> str:
        """Generates an OR function string, e.g., 'var1 ∨ var2'."""
        return " " + " ∨ ".join(map(str, output_variables)) + " "

    def to_cnf(self, output_variables: List[int], coupling_variable: int) -> List[List[int]]:
        """
        Converts the OR logic to CNF.
        C <=> (V1 ∨ V2 ∨ ... ∨ Vn) becomes:
        (¬V1 ∨ C) ∧ (¬V2 ∨ C) ∧ ... ∧ (¬Vn ∨ C) ∧ (V1 ∨ V2 ∨ ... ∨ Vn ∨ ¬C)
        """
        clauses = []
        # Add clauses (¬Vi ∨ C) for each output variable Vi
        for var in output_variables:
            clauses.append([-var, coupling_variable])

        # Add the clause (V1 ∨ V2 ∨ ... ∨ Vn ∨ ¬C)
        clauses.append(output_variables + [-coupling_variable])

        return clauses

class AndCoupling(CouplingStrategy):
    """
    Represents a conjunctive (AND) coupling between local networks.

    The coupling signal is active only if all of the output variables are active.
    """

    def generate_coupling_function(self, output_variables: List[int]) -> str:
        """Generates an AND function string, e.g., 'var1 ∧ var2'."""
        return " " + " ∧ ".join(map(str, output_variables)) + " "

    def to_cnf(self, output_variables: List[int], coupling_variable: int) -> List[List[int]]:
        """
        Converts the AND logic to CNF.
        C <=> (V1 ∧ V2 ∧ ... ∧ Vn) becomes:
        (V1 ∨ ¬C) ∧ (V2 ∨ ¬C) ∧ ... ∧ (Vn ∨ ¬C) ∧ (¬V1 ∨ ¬V2 ∨ ... ∨ ¬Vn ∨ C)
        """
        clauses = []
        # Add clauses (Vi ∨ ¬C) for each output variable Vi
        for var in output_variables:
            clauses.append([var, -coupling_variable])

        # Add the clause (¬V1 ∨ ¬V2 ∨ ... ∨ ¬Vn ∨ C)
        negated_vars = [-var for var in output_variables]
        clauses.append(negated_vars + [coupling_variable])

        return clauses

class ThresholdCoupling(CouplingStrategy):
    """
    Represents a threshold coupling.

    The coupling signal is active if at least `k` of the output variables are active.
    """
    def __init__(self, threshold: int):
        if threshold <= 0:
            raise ValueError("Threshold must be a positive integer.")
        self.threshold = threshold

    def generate_coupling_function(self, output_variables: List[int]) -> str:
        """Generates a threshold function string, e.g., 'Threshold(2, {var1, var2})'."""
        return f" Threshold({self.threshold}, {{{', '.join(map(str, output_variables))}}}) "

    def to_cnf(self, output_variables: List[int], coupling_variable: int) -> List[List[int]]:
        """
        Converts the threshold logic "at least k out of n" to CNF.
        C <=> (sum(Vi) >= k)

        Warning: This method uses itertools.combinations to generate the clauses,
        which can be computationally expensive and lead to a combinatorial explosion
        in the number of clauses for a large number of output variables.
        """
        n = len(output_variables)
        k = self.threshold
        clauses = []

        if k > n:
            # The threshold can never be met, so C must be false.
            return [[-coupling_variable]]

        # Implication 1: (sum(Vi) >= k) => C
        # If at least k variables are true, C must be true.
        # This is equivalent to: if any n-k+1 variables are false, C is false.
        # Or: for any k variables, if all are true, C must be true.
        # Clause: (¬V_i1 ∨ ¬V_i2 ∨ ... ∨ ¬V_ik ∨ C) for all combinations of k variables.
        for combo in combinations(output_variables, k):
            clauses.append([-v for v in combo] + [coupling_variable])

        # Implication 2: C => (sum(Vi) >= k)
        # If C is true, at least k variables must be true.
        # This is equivalent to: if C is true, you cannot have n-k+1 variables that are false.
        # Clause: (¬C ∨ V_i1 ∨ V_i2 ∨ ... ∨ V_i(n-k+1)) for all combinations of n-k+1 variables.
        for combo in combinations(output_variables, n - k + 1):
            clauses.append([-coupling_variable] + list(combo))

        return clauses
