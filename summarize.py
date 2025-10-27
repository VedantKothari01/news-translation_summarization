import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings('ignore')

class SummarizationPipeline:
    def __init__(self):
        self.model_name = "facebook/bart-large-cnn"
        self.model = None
        self.tokenizer = None
    
    def summarize(self, text, max_length=300, min_length=80, num_beams=4):
        if self.model is None:
            self._load_model()
        
        inputs = self.tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=1.5,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    
    def _load_model(self):
        print("Loading BART model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded")
