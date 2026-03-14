import os
from dotenv import load_dotenv
from pinecone import Pinecone

load_dotenv()

pinecone_api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=pinecone_api_key)

# create index
index_name = "travel-planner-index"
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "llama-text-embed-v2",
            "field_map": {"text": "chunk_text"}
        }
    )
