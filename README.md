Retrieval-Augmented Generation (RAG) Question Answering System
This project implements a RAG system to answer questions based on three AI research papers. It processes PDFs, retrieves relevant chunks, and generates answers with source attribution.
Setup Instructions

Clone the repository:
git clone https://github.com/Sri-Annaamalai-M/RAG-system-for-Q-A.git



Create and activate a virtual environment:
python3 -m venv venv
source venv/bin/activate


Install dependencies:
pip install -r requirements.txt
pip install faiss-cpu


Download the research papers and place them in data/raw/:

Attention is All You Need
Retrieval-Augmented Generation
Language Models are Few-Shot Learners


Run the project:
chmod +x run.sh
./run.sh



Usage
Run the main script to preprocess documents and start the QA system:
python src/main.py

Follow the prompts to enter questions. Example questions:

"What are the main components of a RAG model?"
"What are the two sub-layers in each encoder layer of the Transformer model?"

Project Structure

data/: Stores raw PDFs and processed chunks/embeddings.
src/: Python scripts for document processing, retrieval, and generation.
tests/: Unit tests.
requirements.txt: Python dependencies.
config.yaml: Configuration settings.
run.sh: Setup and run script.

Requirements

Python 3.8+
FAISS (CPU or GPU version)
NVIDIA GPU (optional, for faster inference)

License
MIT License