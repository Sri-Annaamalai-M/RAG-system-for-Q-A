#!/bin/bash

# Set up virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install faiss-cpu

# Download NLTK data
python -m nltk.downloader punkt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the main script
python src/main.py

# Deactivate virtual environment
deactivate