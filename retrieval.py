import chromadb

def retrieve_via_query(query: str, top_k: int = 5) -> list:
    """Retrieve relevant documents from the vector database based on the query."""
    
    db_path = "./chroma_data"
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(name="documents_primary")
    
    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )
    
    retrieved_docs = []
    for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
        retrieved_docs.append({
            "document": doc,
            "metadata": metadata
        })
    
    return retrieved_docs