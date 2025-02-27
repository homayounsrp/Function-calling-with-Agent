from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.embeddings import AzureOpenAIEmbeddings
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def llm_setup(mode="generate"):

    load_dotenv()

    # Ensure Azure environment variables are loaded
    os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")
    os.environ["AZURE_OPENAI_ENDPOINT"] = os.getenv("AZURE_OPENAI_ENDPOINT")
    
    # Initialize the Azure generation model
    llm = AzureChatOpenAI(
        azure_deployment='gpt-4o-mini',
        api_version="2024-02-15-preview",
        temperature=0
    )
    # Also initialize the Azure embeddings model
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="gpt-4o-mini",
        model="text-embedding-ada-002",
        chunk_size=1000
    )
    if mode == "generate":
        return llm
    elif mode == "embed":
        return embeddings
    else:
        raise ValueError("Invalid mode. Choose 'generate' or 'embed'.")





def generate_llm_response(prompt, context, llm):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that provides detailed and accurate answers based on the given context."),
        ("human", "Context: {context}\nPrompt: {prompt}\nAnswer:")
    ])
    chain = RunnablePassthrough.assign(answer={
        "prompt": lambda x: x["prompt"],
        "context": lambda x: x["context"]
    } | prompt_template | llm)
    return chain.invoke({"prompt": prompt, "context": context})
