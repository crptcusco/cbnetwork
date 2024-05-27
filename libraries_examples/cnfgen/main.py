import cnfgen

F = cnfgen.CNF()

F.add_clause([1, -2, 3])
F.add_clause([-1, -3])
F.add_clause([2, -3])
F.add_clause([2])

# outputs a pair
outcome, assignment = F.solve()
print(outcome)
print(assignment)
print(F.to_dimacs())
print(F.to_latex())

bool = F.is_satisfiable(cmd='minisat -no-pre')
# bool = F.is_satisfiable(cmd='glucose -pre')
# bool = F.is_satisfiable(cmd='lingeling --plain')
# bool = F.is_satisfiable(cmd='sat4j')
# bool = F.is_satisfiable(cmd='my-hacked-minisat -pre',sameas='minisat')
# bool = F.is_satisfiable(cmd='patched-lingeling',sameas='lingeling')

print(bool)

# # generate k random networks
# F = cnfgen.RandomKCNF(3, 6, 2, seed=None, planted_assignments=[1, -2, 3])
# print(F.to_dimacs())
#
# bool = F.is_satisfiable(cmd='minisat -no-pre')


# Planted assignments in dictionary form
planted_assignments = {1: True, 2: False, 3: True}

# Convert the dictionary to the format expected by sample_clauses
planted_assignments_list = [var if val else -var for var, val in planted_assignments.items()]
print(planted_assignments_list)
# Generate a random 3-CNF formula with 6 variables and 2 clauses
F = cnfgen.RandomKCNF(3, 6, 2, seed=None, planted_assignments=[[5, 2], [-1, 3]])

# Generate a random 3 CNF formula with 6 variables and 2 clauses
F = cnfgen.RandomKCNF(3, 6, 2, seed=None, planted_assignments=[[5, 2], [-1, 3]])

# Print the formula in DIMACS format
print(F.to_dimacs())

# Check if the formula is satisfiable using minisat
is_satisfiable = F.is_satisfiable(cmd='minisat -no-pre')
print("Is the formula satisfiable?", is_satisfiable)
