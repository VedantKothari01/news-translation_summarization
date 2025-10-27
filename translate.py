import torch
import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import warnings
warnings.filterwarnings('ignore')

class TranslationPipeline:
    def __init__(self):
        self.model_name = "facebook/mbart-large-50-many-to-many-mmt"
        self.model = None
        self.tokenizer = None
        
        self.lang_codes = {
            'en': 'en_XX', 'hi': 'hi_IN', 'es': 'es_XX', 'fr': 'fr_XX',
            'de': 'de_DE', 'zh': 'zh_CN', 'ar': 'ar_AR', 'ja': 'ja_XX',
            'pt': 'pt_XX', 'ru': 'ru_RU', 'ta': 'ta_IN', 'te': 'te_IN',
            'mr': 'mr_IN', 'gu': 'gu_IN', 'bn': 'bn_IN'
        }
    
    def translate(self, text, source_lang, target_lang, max_length=512):
        if source_lang == target_lang:
            return text
        
        if self.model is None:
            self._load_model()
        
        src_code = self.lang_codes.get(source_lang, 'en_XX')
        tgt_code = self.lang_codes.get(target_lang, 'en_XX')
        
        self.tokenizer.src_lang = src_code
        
        # Don't truncate - translate full text
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            generated = self.model.generate(
                **encoded,
                forced_bos_token_id=self.tokenizer.lang_code_to_id[tgt_code],
                num_beams=4,
                max_length=max_length,
                early_stopping=False,
                no_repeat_ngram_size=2
            )
        
        result = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return result
    
    def _load_model(self):
        print("Loading mBART-50 model...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name)
        self.model = MBartForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded")
