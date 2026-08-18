from langchain_groq import ChatGroq

import os
os.environ["GROQ_API_KEY"] = "key"


from groq import Groq
client = Groq()
print(client.models.list())
