import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
print(f"Checking index: {index_name}")
try:
    index = pc.Index(index_name)
    stats = index.describe_index_stats()
    print(f"Stats: {stats}")
except Exception as e:
    print(f"Error: {e}")
