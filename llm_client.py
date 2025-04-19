from langchain_openai import ChatOpenAI
import os

def get_gpt_response(prompt):
    llm = ChatOpenAI(
        model="gpt-4.1-nano",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    response = llm.invoke(prompt)
    return response.content
