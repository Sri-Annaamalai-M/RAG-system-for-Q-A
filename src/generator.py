import os
import yaml
import google.generativeai as genai

class Generator:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        # Configure Gemini API
        api_key = self.config['models'].get('gemini_api_key') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("Gemini API key not found in config.yaml or GOOGLE_API_KEY environment variable")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.config['models']['generator_model'])
        
    def generate_answer(self, query, chunks, metadata):
        # Create a prompt with the query and retrieved chunks
        prompt = f"Answer the following question based on the provided context. Provide a concise and accurate response, and avoid adding information not present in the context.\n\nQuestion: {query}\n\nContext:\n"
        for i, chunk in enumerate(chunks, 1):
            prompt += f"Chunk {i}: {chunk}\n\n"
        
        # Call Gemini API
        response = self.model.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": 150,
                "temperature": 0.7,
            }
        )
        
        # Extract answer (assuming response.text for Gemini API)
        answer = response.text
        
        # Add citations
        citations = [f"{meta['paper_title']}, Chunk ID: {meta['chunk_id']}" for meta in metadata]
        answer += f"\n\nSources:\n" + "\n".join([f"- {cite}" for cite in citations])
        return answer