from langchain_groq import ChatGroq

llm = ChatGroq(
    api_key="key",
    model_name="groq/compound",
    temperature=0
)

response = llm.invoke("Hello Groq!")
print(response.content)
