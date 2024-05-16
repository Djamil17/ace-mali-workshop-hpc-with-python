import ast


def create_function_from_string():
    # Get a string from the user
    function_string = input("Enter the function body: ")

    # Parse the function string to extract its arguments
    function_ast = ast.parse(function_string)
    parameters = []
    for node in ast.walk(function_ast):
        if isinstance(node, ast.Name):
            parameters.append(node.id)
            break

    # Define the function dynamically
    def dynamic_function(*args):
        # Use ast.literal_eval() to safely evaluate the function string
        function_body = ast.literal_eval(function_string)

        # Execute the function body with the given arguments
        exec(compile(function_ast, filename="<ast>", mode="exec"))

        return function_body(*args)

    return dynamic_function, parameters


# Create a function based on user input
user_function, parameters = create_function_from_string()
print(user_function)
print(parameters)
# Now you can call the dynamically created function
args = [int(input(f"Enter value for parameter '{param}': ")) for param in parameters]
result = user_function(*args)
print("Result:", result)
