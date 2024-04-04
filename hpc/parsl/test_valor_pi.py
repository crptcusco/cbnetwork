# -*- coding: utf-8 -*-

import parsl
from parsl.app.app import python_app
import random

# Configure Parsl
parsl.load()


@python_app
def monte_carlo_pi(num_samples):
    count_inside_circle = 0

    for _ in range(num_samples):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)

        # Check if the point is inside the unit circle
        if x ** 2 + y ** 2 <= 1:
            count_inside_circle += 1

    return count_inside_circle


def estimate_pi(num_samples, num_tasks):
    # Calculate the number of samples per task
    samples_per_task = num_samples // num_tasks

    # Launch parallel tasks to perform the estimation
    results = [monte_carlo_pi(samples_per_task).result() for _ in range(num_tasks)]

    # Sum the results from all tasks
    total_inside_circle = sum(results)

    # Calculate the estimate of pi
    pi_estimate = (total_inside_circle / num_samples) * 4

    return pi_estimate


# Total number of samples
total_samples = 100000000

# Number of parallel tasks
num_tasks = 4

# Estimate the value of pi
pi_value = estimate_pi(total_samples, num_tasks)

print("Estimated value of pi: {}".format(pi_value))

# Close Parsl
parsl.clear()

# Execute code in Slurm
# srun -N 3 -n 3 -c 1 python test_valor_pi.py
