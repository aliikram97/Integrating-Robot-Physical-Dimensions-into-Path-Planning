import ast
import cv2
import numpy as np

def generate_flowchart(code):
    # Parse the Python code into an abstract syntax tree (AST)
    parsed_code = ast.parse(code)

    # Basic representation of the code structure
    flowchart_representation = f"Python Code:\n{code}\n\nAbstract Syntax Tree:\n{parsed_code}"

    # Convert the representation to an image using OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = np.ones((500, 800, 3), np.uint8) * 255  # Create a white canvas

    # Add the text to the image
    lines = flowchart_representation.split('\n')
    for i, line in enumerate(lines):
        cv2.putText(image, line, (10, 30 * (i + 1)), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    # Display the image using OpenCV
    cv2.imshow('Flowchart', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example Python code
python_code = """
def calculate_sum(a, b):
    result = a + b
    return result

x = 5
y = 10
total = calculate_sum(x, y)
print(total)
"""

# Generate a basic flowchart-like representation for the provided code
generate_flowchart(python_code)
