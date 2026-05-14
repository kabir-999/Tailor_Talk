import os
import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")

if index_name not in [idx.name for idx in pc.list_indexes()]:
    print(f"Creating index {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=1024, # llama-text-embed-v2
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    print("Index ready!")
else:
    print(f"Index {index_name} already exists.")
