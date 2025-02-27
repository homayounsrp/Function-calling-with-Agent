def handle_multiply(num1: float, num2: float) -> str:
    """
    Handle multiplication-related queries. Takes two numbers as separate inputs.
    """
    try:
        result = num1 * num2
        return f"Result of multiplying {num1} by {num2} is: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

def handle_divide(num1: float, num2: float) -> str:
    """
    Handle division-related queries. Takes two numbers as separate inputs.
    """
    try:
        if num2 == 0:
            return "Division by zero is not allowed."
        result = num1 / num2
        return f"Result of dividing {num1} by {num2} is: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
