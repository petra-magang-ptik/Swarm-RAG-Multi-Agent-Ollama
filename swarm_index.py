from swarm import Swarm, Agent, Result
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.fastembed import FastEmbedEmbedding
import os
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# load_dotenv()
model = os.getenv('LLM_MODEL', 'llama3.2:latest')
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",  
    api_key="ollama"
)
Settings.llm = Ollama(model="llama3.2:latest", base_url="http://127.0.0.1:11434")
###if u want to use the model that implant on the ollama system###
Settings.embed_model = OllamaEmbedding(base_url="http://127.0.0.1:11434", model_name="nomic-embed-text:latest")
###if u want to use the fastembed it took a long time###
# embedding_model="intfloat/multilingual-e5-large"
# Settings.embed_model = FastEmbedEmbedding(model_name=embedding_model, cache_dir="./fastembed_cache")
# Swarm and agent setup
client = Swarm(client=ollama_client)

def create_rag_index(pdf_filepath="docs"):
    documents = SimpleDirectoryReader(pdf_filepath).load_data()
    index = VectorStoreIndex.from_documents(documents)
    print("Index created.")
    return index

# Create the RAG index
rag_index = create_rag_index()

def query_rag(query_str):
    query_engine = rag_index.as_query_engine()
    response = query_engine.query(query_str)
    return str(response)

# Define agents
def triage_agent_instructions(context_variables):
    return """You are a triage agent.
    If the user asks a question related to the document, hand off to the RAG agent.
    """

def rag_agent_instructions(context_variables):
    return """You are a RAG agent. Answer user questions by using the `query_rag` function to retrieve information.
    """
def handoff_to_rag_agent(*args, **kwargs):
    return Result(agent=rag_agent)

triage_agent = Agent(
    name="Triage Agent",
    instructions=triage_agent_instructions,
    functions=[handoff_to_rag_agent],
    model=model
)

rag_agent = Agent(
    name="RAG Agent",
    instructions=rag_agent_instructions,
    functions=[query_rag],
    model=model
)
# Run the Swarm App
def run_swarm_app():
    print("Welcome to the RAG Swarm App!")
    current_agent = triage_agent
    messages = []
    while True:
        user_input = input("User: ")
        if user_input.lower() == "exit":
            break
        messages.append({"role": "user", "content": user_input})
        response = client.run(
            agent=current_agent,
            messages=messages,
        )
        print(f"{response.agent.name}: {response.messages[-1]['content']}")
        messages = response.messages
        current_agent = response.agent

if __name__ == "__main__":
    run_swarm_app()
