# Retrieval-Augmented Generation (RAG) Question Answering System

This project implements a RAG system to answer questions based on the content of three AI research papers. It processes PDFs, retrieves relevant text chunks, and generates answers with source attribution using Gemini 2.0 Flash.

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
git clone https://github.com/Sri-Annaamalai-M/RAG-system-for-Q-A.git
cd RAG-system-for-Q-A
   ```

2. **Create and activate a virtual environment:**
   ```bash
python3 -m venv venv
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
pip install -r requirements.txt
pip install faiss-cpu
   ```

4. **Download NLTK and spaCy data/models:**
   ```bash
python -m nltk.downloader punkt
python -m spacy download en_core_web_sm
   ```
   Or simply run the provided script:
   ```bash
chmod +x run.sh
./run.sh
   ```

5. **Download the research papers and place them in `data/raw/`:**
   - Attention is All You Need
   - Retrieval-Augmented Generation
   - Language Models are Few-Shot Learners

---

## Usage

Run the main script to preprocess documents and start the QA system:
```bash
python src/main.py
```

- The system will:
  1. **Process PDFs** in `data/raw/`, chunk and embed their content, and build a FAISS index.
  2. **Start an interactive QA loop**. Enter your question at the prompt (type `quit` to exit).
  3. **Retrieve relevant chunks** and generate an answer using Gemini 2.0 Flash, with sources cited.

**Example questions:**
- What are the main components of a RAG model?
- What are the two sub-layers in each encoder layer of the Transformer model?

---

## Project Structure

- `data/` - Stores raw PDFs and processed chunks/embeddings. **(Ignored by git)**
- `src/` - Python scripts for document processing, retrieval, and generation.
- `tests/` - Unit tests.
- `requirements.txt` - Python dependencies.
- `config.yaml` - Configuration settings.
- `run.sh` - Setup and run script.
- `.gitignore` - Excludes data, venv, outputs, and sensitive files from version control.

---

## Notes on Version Control
- The `.gitignore` file ensures that large data files, processed outputs, virtual environments, and sensitive information (such as API keys and environment variables) are **not** tracked by git.
- If you want to keep sample data or configuration, add example files (e.g., `config.example.yaml`).

---

## Requirements
- Python 3.8+
- FAISS (CPU or GPU version)
- NVIDIA GPU (optional, for faster inference)

---

## License
MIT License