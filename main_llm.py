import os
import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# 1️⃣ Set your API key
os.environ["GROQ_API_KEY"] = "key"

# 2️⃣ Create LLM (switch to OpenAI GPT OSS 120B)
llm = ChatGroq(
    temperature=0,
    model_name="openai/gpt-oss-120b"
)

# 3️⃣ Create Chroma collection with Hugging Face embeddings
embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
client = chromadb.Client()
collection = client.get_or_create_collection(
    "jobs_collection",
    embedding_function=embedding_fn
)

# 4️⃣ Add portfolio data (run once)
collection.add(
    documents=[
        "Machine learning and Python AI solutions",
        "WordPress website development services",
        "Magento e-commerce platform development"
    ],
    metadatas=[
        {"links": "https://example.com/ml-python-portfolio"},
        {"links": "https://example.com/wordpress-portfolio"},
        {"links": "https://example.com/magento-portfolio"}
    ],
    ids=["doc1", "doc2", "doc3"]
)

# 5️⃣ Job data
json_res = [
    {
        "title": "AI Engineer",
        "skills": "Python, Machine Learning, NLP, APIs",
        "description": "Hiring AI engineer to build ML models and NLP systems."
    }
]

job = json_res[0]

# 6️⃣ Query links
raw_links = collection.query(query_texts=[job["skills"]], n_results=2).get("metadatas", [])

# 7️⃣ Clean links
clean_links = [item["links"] for group in raw_links for item in group]
unique_links = list(set(clean_links))

# 8️⃣ Prompt
prompt_email = PromptTemplate.from_template(
    """
    ### JOB DESCRIPTION:
    {job_description}
   
    ### INSTRUCTION:
    You are Mohan, a business development executive at AtliQ.
    AtliQ is an AI & Software Consulting company dedicated to facilitating
    the seamless integration of business processes through automated tools.
    Over our experience, we have empowered numerous enterprises
    with tailored solutions, fostering scalability,
    process optimization, cost reduction, and heightened overall efficiency.
    Your job is to write a cold email to the client regarding the
    job mentioned above describing the capability of AtliQ
    in fulfilling their needs.
    Also add the most relevant ones from the following links
    to showcase Atliq's portfolio: {link_list}
    Remember you are Mohan, BDE at AtliQ.
    Do not provide a preamble.
    ### EMAIL (NO PREAMBLE):
    """
)

# 9️⃣ Chain + Invoke
chain_email = prompt_email | llm
res = chain_email.invoke({
    "job_description": str(job),
    "link_list": unique_links
})
print(res.content)
