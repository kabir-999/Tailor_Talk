import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
print(f"Indexes: {[idx.name for idx in pc.list_indexes()]}")
