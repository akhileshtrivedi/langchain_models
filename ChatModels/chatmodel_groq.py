from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv


load_dotenv()

llm = ChatGroq(
        model_name="llama3-8b-8192", 
        temperature=0
    )

prompt = ChatPromptTemplate.from_messages([
        ("human", "Tell me a fun fact about {topic}.")
    ])

chain = prompt | llm 

response = chain.invoke({"topic": "cats"})
print(response.content)

result = llm.invoke("What is the capital of India")

print(result)