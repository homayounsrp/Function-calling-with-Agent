from langchain_core.prompts import ChatPromptTemplate
from typing import Any, Dict
from LLM.output_parser import *
from LLM.load_llm import generate_llm_response
from LLM.load_llm import llm_setup


# For text generation using Azure
llm_model = llm_setup(mode="generate")


def llm_router(sub_query_text: str, generate_llm) -> str:
    """
    Use an LLM to decide which math operation should process the sub-query.

    Args:
        sub_query_text (str): The sub-query text.
        generate_llm: A callable that takes prompt messages and returns an LLM response.

    Returns:
        str: One of "multiply" or "divide".
    """
    router_prompt = (
        "You are a math query router. Analyze the following sub-query and decide which mathematical operation should process it.\n"
        "Your options are:\n"
        "- 'multiply' for queries that require multiplication,\n"
        "- 'divide' for queries that require division.\n"
        "Return only one word: multiply or divide.\n\n"
        "Sub-query: {sub_query_text}"
    )
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a math routing assistant."),
        ("human", router_prompt)
    ])
    
    messages = prompt_template.format_prompt(sub_query_text=sub_query_text).to_messages()
    routing_response = generate_llm(messages)
    
    # Support different response formats
    if hasattr(routing_response, "choices"):
        decision = routing_response.choices[0].message.content.strip().lower()
    elif hasattr(routing_response, "content"):
        decision = routing_response.content.strip().lower()
    else:
        raise ValueError("Unexpected LLM response format")
    
    return decision


def route_sub_query(sub_query: Dict[str, Any],
                    generate_llm,
                    tool_functions) -> Any:
    """
    Route the sub-query to the appropriate mathematical operation based on the detected math type.

    Args:
        sub_query (dict): The sub-query data containing question, number1, and number2.
        generate_llm: Callable for LLM operations.
        tool_functions: An object containing the tool functions.

    Returns:
        Any: The result of the appropriate function call.
    """
    route_decision = llm_router(sub_query["question"], generate_llm)
    print(f"LLM Router decision: {route_decision} for question: {sub_query['question']}")
    
    # Extract numbers from the query
    num1 = sub_query.get("number1")
    num2 = sub_query.get("number2")

    # Ensure both numbers are provided
    if num1 is None or num2 is None:
        raise ValueError("Both number1 and number2 must be provided.")

    # Convert numbers to float for calculation
    try:
        num1 = float(num1)
        num2 = float(num2)
    except ValueError:
        raise ValueError("Invalid number format. Ensure both inputs are valid numbers.")

    if route_decision == "multiply":
        return tool_functions.handle_multiply(num1, num2)
    elif route_decision == "divide":
        return tool_functions.handle_divide(num1, num2)
    else:
        raise ValueError(f"Unexpected route decision: {route_decision}")



def combine_responses(main_query, sub_query_results: Dict[int, Any], generate_llm) -> str:
    """
    Combine the responses from multiple sub-queries into one final comprehensive answer.
    
    Args:
        sub_query_results: Dictionary mapping sub-query IDs to their responses.
        generate_llm: Callable for LLM operations.
        
    Returns:
        str: A single, refined answer (only the content) that combines all sub-query responses,
             formatted in markdown.
    """
    def extract_text(response):
        return response.content if hasattr(response, "content") else response

    combined_text = "\n\n".join(
        f"Response {key}: {extract_text(result)}"
        for key, result in sub_query_results.items()
    )
    
    combine_prompt = (
        "The following are responses from multiple sub-queries:\n\n"
        f"{combined_text}\n\n"
        f"Combine and refine these responses into one comprehensive, cohesive final answer in response to the user query: {main_query}.\n"
        "Present your answer in clear, concise language and in markdown format. "
        "If any sub-queries are empty, mention that the response is based on the limited information provided."
    )
    
    final_response = generate_llm_response(prompt=combine_prompt, context="", llm=llm_model)
    # Return only the content of the final response.
    return final_response['answer'].content


def process_query_plan(main_query, query_plan: Any,
                       generate_llm,
                       tool_functions) -> str:
    """
    Process the query plan by routing each sub-query to the appropriate math operation handler,
    then combine the individual responses into one final answer.
    
    Args:
        query_plan: Either a dict with keys "query" and "answer" (where "answer" holds the QueryPlan)
                    or a QueryPlan instance.
        generate_llm: Callable for LLM operations.
        tool_functions: An object or module containing the tool functions.
        
    Returns:
        str: The final combined response.
    """
    if isinstance(query_plan, dict):
        query_plan = query_plan.get("answer")
        if query_plan is None:
            raise ValueError("The provided dictionary does not contain an 'answer' key with the query plan.")
    
    results = {}
    for sub_query in query_plan.query_graph:
        sub_query_dict = sub_query.dict()
        result = route_sub_query(sub_query_dict, generate_llm, tool_functions)
        results[sub_query.id] = result
    
    # Combine individual responses into one final answer.
    final_response = combine_responses(main_query, results, generate_llm)
    return final_response
