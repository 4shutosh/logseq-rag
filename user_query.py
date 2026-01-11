## no chunking and embedding here, just user ops

from retrieval import retrieve_via_query
def retrieve_documents(query: str, top_k: int =5) -> list:
    return retrieve_via_query(query, top_k)


from prompt import get_LLM_response
# main function
if __name__ == "__main__":
    # retrieve via a sample query
    sample_query = "best strategy"
    retrieved_docs = retrieve_documents(sample_query, top_k=3)

    llm_response = get_LLM_response(sample_query, retrieved_docs)
    print("\n----- LLM RESPONSE -----")
    print(llm_response)