import chromadb
import os

def store_chunked_data_in_db(chunks: list):
    """Store chunked documents in the vector database."""

    # Create a persistent client that stores data locally
    db_path = "./chroma_data"
    os.makedirs(db_path, exist_ok=True)
    
    client = chromadb.PersistentClient(path=db_path)

    collection = client.get_or_create_collection(name="documents_primary")

    for doc in chunks:
        collection.add(
            documents=[doc.page_content],
            metadatas=[doc.metadata],
            ids=[str(doc.metadata["chunk_id"])]
        )

    print(f"Stored {len(chunks)} chunks in the database.")