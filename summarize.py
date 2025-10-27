import requests
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

class SummarizationPipeline:
    def __init__(self):
        self.api_key = os.getenv('HUGGINGFACE_API_KEY', '')
        self.base_url = "https://api-inference.huggingface.co/models"
        self.model_name = "facebook/bart-large-cnn"
    
    def summarize(self, text, max_length=200, min_length=50, num_beams=4):
        if not self.api_key:
            return text[:max_length] + "..."
        
        if len(text) < 100:
            return text
        
        chunks = self._chunk_text(text)
        chunk_summaries = []
        
        for chunk in chunks:
            try:
                summary = self._summarize_chunk(chunk)
                if summary and len(summary) > 20:
                    chunk_summaries.append(summary)
            except:
                pass
        
        if not chunk_summaries:
            return text[:max_length] + "..."
        
        combined = " ".join(chunk_summaries)
        
        if len(combined) > max_length:
            return combined[:max_length] + "..."
        
        return combined
    
    def _chunk_text(self, text):
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 1000:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks[:5]
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _summarize_chunk(self, text):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        response = requests.post(
            f"{self.base_url}/{self.model_name}",
            json={"inputs": text},
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and result:
                return result[0].get('summary_text', '')
            elif isinstance(result, dict):
                return result.get('summary_text', '')
        
        raise Exception(f"Summarization failed: {response.status_code}")
