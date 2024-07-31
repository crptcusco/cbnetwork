import random


class CNFList:
    @staticmethod
    def generate_cnf(l_inter_vars, input_coup_sig_index, max_clauses=2, max_literals=3):
        """
        Generates a CNF (Conjunctive Normal Form) list with random clauses.

        Args:
            l_inter_vars (list): List of internal variables.
            input_coup_sig_index (int): Index of the input coupling signal.
            max_clauses (int): Maximum number of clauses to generate.
            max_literals (int): Maximum number of literals per clause.

        Returns:
            list: List of clauses in CNF format.
        """
        num_clauses = random.randint(1, max_clauses)  # Ensure at least one clause is generated
        l_cnf = []

        # Generate the clause for external signals
        if input_coup_sig_index is not None:
            var = input_coup_sig_index
            if random.choice([True, False]):
                var = -var
            l_cnf.append([var])

        for _ in range(num_clauses):
            clause = []
            while len(clause) < max_literals:
                var = random.choice(l_inter_vars)
                if var != input_coup_sig_index and -var != input_coup_sig_index:
                    if random.choice([True, False]):
                        var = -var
                    clause.append(var)

            # Remove redundant literals within the clause
            clause = CNFList.simplify_clause(clause)

            # Ensure the clause is not empty and has at least one literal
            if clause:
                l_cnf.append(clause)

        # Remove empty clauses
        l_cnf = [clause for clause in l_cnf if clause]

        # Remove duplicate clauses
        l_cnf = CNFList.remove_duplicates(l_cnf)

        # Ensure there's at least one non-empty clause
        if not l_cnf:
            # If all generated clauses are empty, add at least one valid clause
            clause = []
            while len(clause) < max_literals:
                var = random.choice(l_inter_vars)
                if random.choice([True, False]):
                    var = -var
                clause.append(var)
            l_cnf.append(clause)

        return l_cnf

    @staticmethod
    def simplify_clause(clause):
        """
        Simplifies a clause by removing duplicate and complementary literals.

        Args:
            clause (list): List of literals in a clause.

        Returns:
            list: Simplified clause.
        """
        # Remove duplicate literals
        clause = list(set(clause))

        # Check for complementary literals (e.g., x and -x) and remove both
        simplified_clause = []
        for literal in clause:
            if -literal not in clause:
                simplified_clause.append(literal)

        return simplified_clause

    @staticmethod
    def remove_duplicates(l_cnf):
        """
        Removes duplicate clauses from the CNF list.

        Args:
            l_cnf (list): List of clauses in CNF format.

        Returns:
            list: List of unique clauses.
        """
        # Convert each clause to a tuple and create a set to remove duplicates
        unique_clauses = set(tuple(sorted(clause)) for clause in l_cnf)
        # Convert the unique tuples back to lists
        return [list(clause) for clause in unique_clauses]

