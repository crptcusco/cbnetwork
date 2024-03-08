import parsl
from parsl.app.app import python_app
import random

# Configura parsl
parsl.load()

@python_app
def monte_carlo_pi(num_samples):
    count_inside_circle = 0

    for _ in range(num_samples):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        # Verifica si el punto está dentro del círculo unitario
        if x**2 + y**2 <= 1:
            count_inside_circle += 1

    return count_inside_circle

def estimate_pi(num_samples, num_tasks):
    # Calcula el número de muestras por tarea
    samples_per_task = num_samples // num_tasks

    # Lanza tareas paralelas para realizar la estimación
    results = [monte_carlo_pi(samples_per_task).result() for _ in range(num_tasks)]

    # Suma los resultados de todas las tareas
    total_inside_circle = sum(results)

    # Calcula la estimación de pi
    pi_estimate = (total_inside_circle / num_samples) * 4

    return pi_estimate

# Número total de muestras
total_samples = 1000000

# Número de tareas paralelas
num_tasks = 4

# Estima el valor de pi
pi_value = estimate_pi(total_samples, num_tasks)

print(f"Estimación de pi: {pi_value}")

# Cerrar parsl
parsl.clear()

# Execute code in Sdumont
# srun -N 3 -n 3 -c 1 python test_valor_pi.py
