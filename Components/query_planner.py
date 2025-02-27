from LLM.output_parser import *
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

def plan_query(complex_query: str, generate_llm) -> QueryPlan:
    """
    Generate a structured query plan for a complex query.

    Args:
        complex_query (str): The complex query to be broken down.
        generate_llm: A callable LLM function (e.g., from LangChain) that supports the method 
                      `.with_structured_output(...)` to output a structured result.
    """
    query_plan_prompt = (
        """
        Given the user's input prompt: {query}
        
        - If the prompt is straightforward, return it as a single query without further breakdown.
        - If the prompt is complex and contains multiple distinct ideas or requirements, decompose it into a series of smaller, focused subqueries.
        """
    )

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at planning multi-step queries."),
        ("human", query_plan_prompt)
    ])

    chain_input_prompt_llm = (
        {"query": lambda x: x["query"]}
        | prompt_template
        | generate_llm.with_structured_output(QueryPlan)
    )

    chain = RunnablePassthrough.assign(answer=chain_input_prompt_llm)
    plan = chain.invoke({"query": complex_query})
    return plan
