from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
import argparse


def perform_fixed_size_chunking(document, chunk_size=10, chunk_overlap=200):
    # Create the text splitter with optimal parameters
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    
    # Split the text into chunks
    chunks = text_splitter.split_text(document)
    print(f"Document split into {len(chunks)} chunks")
    
    # Convert to Document objects with metadata
    documents = []
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={
                "chunk_id": i,
                "total_chunks": len(chunks),
                "chunk_size": len(chunk),
                "chunk_type": "fixed-size"
            }
        )
        documents.append(doc)
    
    return documents


def load_md_as_document(file_path: str) -> Document:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return Document(page_content=content, metadata={"source": file_path})

# Example usage
if __name__ == "__main__":
    
    # Create the dummy document
    document = load_md_as_document("logseq_sample_journal.md").page_content
    
    # Process with fixed-size chunking
    chunked_docs = perform_fixed_size_chunking(
        document,
        chunk_size=200,
        chunk_overlap=50
    )
    
    # Display results
    print("\n----- CHUNKING RESULTS -----")
    print(f"Total chunks: {len(chunked_docs)}")
    
    # Print an example chunk
    print("\n----- EXAMPLE CHUNK -----")
    middle_chunk_idx = len(chunked_docs) // 2
    example_chunk = chunked_docs[middle_chunk_idx]
    print(f"Chunk {middle_chunk_idx} content ({len(example_chunk.page_content)} characters):")
    print("-" * 40)
    # print(example_chunk.page_content)
    print("-" * 40)
    print(f"Metadata: {example_chunk.metadata}")

