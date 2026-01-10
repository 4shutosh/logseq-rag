# main py file to run the RAG process, step by step
import chromadb


# step 1 - read the the data and chunk it
from chunk import perform_fixed_size_chunking, load_md_as_document

def chunk_data() -> list:
    # Load the document
    document = load_md_as_document("logseq_sample_journal.md").page_content
    
    # Perform chunking
    chunked_docs = perform_fixed_size_chunking(
        document,
        chunk_size=100,
        chunk_overlap=20
    )
    
    return chunked_docs


# step 2 - embed and store the chunks in vector db
from embed import store_chunked_data_in_db
def embed_and_store_chunks(chunks: list):
    store_chunked_data_in_db(chunks)


# step 3 - retrieve via query
from retrieval import retrieve_via_query
def retrieve_documents(query: str, top_k: int =5) -> list:
    return retrieve_via_query(query, top_k)

from prompt import get_LLM_response
# main function
if __name__ == "__main__":
    # Chunk the data
    chunks = chunk_data()
    
    # Embed and store the chunks
    embed_and_store_chunks(chunks)

    # retrieve via a sample query
    sample_query = "what did I write about foot sweep"
    results = retrieve_documents(sample_query, top_k=1)

    print("\n----- RETRIEVAL RESULTS -----")
    print(f"Results {len(results)}")

    llm_response = get_LLM_response(sample_query, results)
    print("\n----- LLM RESPONSE -----")
    print(llm_response)