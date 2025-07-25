from document_processor import DocumentProcessor
from retriever import Retriever
from generator import Generator

def main():
    config_path = 'config.yaml'
    
    # Step 1: Process documents
    print("Processing documents...")
    processor = DocumentProcessor(config_path)
    chunks, metadata = processor.process_pdfs()
    embeddings = processor.vectorize_chunks(chunks)
    processor.build_faiss_index(embeddings)
    print("Document processing complete.")
    
    # Step 2: Initialize retriever and generator
    retriever = Retriever(config_path)
    generator = Generator(config_path)
    
    # Step 3: Interactive QA loop
    print("\nWelcome to the RAG QA System (Powered by Gemini 2.0 Flash). Enter your question (or 'quit' to exit):")
    while True:
        query = input("\nQuestion: ")
        if query.lower() == 'quit':
            break
        
        # Retrieve relevant chunks
        chunks, metadata = retriever.retrieve(query)
        
        # Generate answer
        try:
            answer = generator.generate_answer(query, chunks, metadata)
            print("\nAnswer:", answer)
        except Exception as e:
            print(f"\nError generating answer: {e}")

if __name__ == "__main__":
    main()