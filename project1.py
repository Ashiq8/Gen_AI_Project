import os
import chromadb
from PyPDF2 import PdfReader
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

os.environ["GROQ_API_KEY"] = "key"


llm = ChatGroq(
    temperature = 0,
    model_name = "openai/gpt-oss-120b"
)

reader = PdfReader("resume.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text()

chunk_size = 500
chunks = [text[i:i+chunk_size]
        for i in range(0, len(text),chunk_size)]

client = chromadb.Client()
collection = client.get_or_create_collection("documents")

collection.add(
    documents = chunks,
    ids = [f"id{i}" for i in range(len(chunks))]
)

query = "What is this document about?"

results = collection.query(
    query_texts = [query],
    n_results = 3

)

context = " ".join(results["documents"][0])

prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant.

    DOCUMENT CONTEXT:
    {context}

    QUESTION:
    {question}

    Answer the question based only on the document.

    """
)

chain = prompt | llm
response = chain.invoke({
    "context": context,
    "question": query
})

print(response.content)