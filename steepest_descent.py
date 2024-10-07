from tabulate import tabulate
import numpy as np
from scipy.optimize import approx_fprime
import math
import pandas as pd
import os

def steepest_descent(header, functionToEvaluate, initial_value, accepted_tolerance, step_size, is_constant, output_file):
    # Get the function
    if functionToEvaluate == 1:
        function = get_first_function_value
    elif functionToEvaluate == 2:
        function = get_second_function_value
    elif functionToEvaluate == 3:
        function = get_third_function_value
    elif functionToEvaluate == 4:
        function = get_fourth_function_value
    else:
        print("Invalid function number")
        return
    
    # Initialize counter
    iterations_count = 1
    # Initialize data
    data = []
    data.append(header)

    while get_gradient_norm(function, initial_value) > accepted_tolerance:
        data_per_iteration = []
        # Choose α_k > 0
        if not is_constant:
            step_size = 1 / iterations_count
        # Calculate ∇f(x_k)
        gradient = get_gradient(function, initial_value)

        # Set x_k ← x_k −α_k∇f(x_k)
        initial_value = initial_value - (step_size * gradient)

        # Append data
        data_per_iteration.append(iterations_count)
        data_per_iteration.append(initial_value.tolist())
        data_per_iteration.append(gradient)
        data_per_iteration.append(get_gradient_norm(function, initial_value))
        
        data.append(data_per_iteration)
        iterations_count += 1
        # If the number of iterations exceeds 1000, stop
        if iterations_count > 1000:
            break
    # Create a DataFrame from the data
    # Skip the header row for data
    df = pd.DataFrame(data[1:], columns=header)  

    # Export the DataFrame to an Excel file
    df.to_excel(output_file, index=False)

    # Print the table
    table = tabulate(data, headers="firstrow", tablefmt="grid")
    print(table)


def get_gradient(function, initial_value):
    return approx_fprime(initial_value, function)

def get_first_function_value(initial_value):
    return 16 * (initial_value[0]) ** 2 + 4 * (initial_value[1]) ** 2 + 2 * initial_value[0] - 4 * initial_value[1] - 4

def get_second_function_value(initial_value):
    return math.sin(initial_value[0] * initial_value[1]) + 81 * (initial_value[0]) ** 2 + 16 * (initial_value[1]) ** 2 

def get_third_function_value(initial_value):
    return math.sqrt((initial_value[0]) ** 2 - 4 * (initial_value[0])  + (initial_value[1]) ** 2 - 6 * (initial_value[1]) + 13 ) + 4

def get_fourth_function_value(initial_value):
    return (1 - initial_value[0]) ** 2 + 400 * (initial_value[1] - (initial_value[0]) ** 2) ** 2

def get_gradient_norm(function, initial_value):
    return np.linalg.norm(get_gradient(function, initial_value))
    

# Common variables
header = ["número k de la iteración",  "Punto x_k", "∇f", "||∇f||"]
initial_value_template = np.array([1.15, 1.15])
accepted_tolerance = 10e-9
is_constant = True

# Functions and step sizes to evaluate
functions = [1, 2, 3, 4]
step_sizes = [0.0001, 0.001, 0.1, 1]

# Evaluate the steepest descent method for each function and step size
for functionNumber in functions:
    # Create a directory for each function
    directory = f"function_{functionNumber}"
    os.makedirs(directory, exist_ok=True)
    for step_size in step_sizes:
        initial_value = initial_value_template.copy()  # Copy the initial value template
        output_file_name = f"steepest_descent_function_{functionNumber}_stepsize_{step_size}.xlsx"
        output_file = os.path.join(directory, output_file_name)
        steepest_descent(header, functionNumber, initial_value, accepted_tolerance, step_size, is_constant, output_file)

# Evaluate the steepest descent method for each function and step size with variable step size
is_constant = False
directory = "variable_step_size"
os.makedirs(directory, exist_ok=True) 
for functionNumber in functions:
    initial_value = initial_value_template.copy()  # Copy the initial value template
    output_file_name = f"steepest_descent_function_{functionNumber}_variable.xlsx"
    output_file = os.path.join(directory, output_file_name)
    steepest_descent(header, functionNumber, initial_value, accepted_tolerance, 0, is_constant, output_file)

