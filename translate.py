import requests
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

class TranslationPipeline:
    def __init__(self):
        self.api_key = os.getenv('HUGGINGFACE_API_KEY', '')
        self.base_url = "https://api-inference.huggingface.co/models"
        
        self.lang_codes = {
            'en': 'en_XX', 'hi': 'hi_IN', 'es': 'es_XX', 'fr': 'fr_XX',
            'de': 'de_DE', 'zh': 'zh_CN', 'ar': 'ar_AR', 'ja': 'ja_XX',
            'pt': 'pt_XX', 'ru': 'ru_RU', 'ta': 'ta_IN', 'te': 'te_IN',
            'mr': 'mr_IN', 'gu': 'gu_IN', 'bn': 'bn_IN'
        }
    
    def translate(self, text, source_lang, target_lang, max_length=1024):
        if source_lang == target_lang:
            return text
        
        if not self.api_key:
            return f"[Translation unavailable - HF_API_KEY not set] {text[:100]}"
        
        src_code = self.lang_codes.get(source_lang, 'en_XX')
        tgt_code = self.lang_codes.get(target_lang, 'en_XX')
        
        model_name = f"Helsinki-NLP/opus-mt-{src_code.replace('_', '-')}-{tgt_code.replace('_', '-')}"
        
        try:
            result = self._translate_with_api(text, model_name)
            if result:
                return result
        except:
            pass
        
        fallback_model = "facebook/mbart-large-50-many-to-many-mmt"
        try:
            return self._translate_mbart(text, src_code, tgt_code)
        except Exception as e:
            return f"[Translation failed: {e}] {text[:100]}"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _translate_with_api(self, text, model_name):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        response = requests.post(
            f"{self.base_url}/{model_name}",
            json={"inputs": text[:1000]},
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and result:
                return result[0].get('translation_text', '')
            elif isinstance(result, dict):
                return result.get('generated_text', '')
        
        response.raise_for_status()
        return None
    
    def _translate_mbart(self, text, src_code, tgt_code):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        formatted_text = f"{src_code} {text[:1000]}"
        
        response = requests.post(
            f"{self.base_url}/facebook/mbart-large-50-many-to-many-mmt",
            json={"inputs": formatted_text},
            headers=headers,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and result:
                translated = result[0].get('generated_text', '')
                return translated.replace(tgt_code, '').strip()
        
        raise Exception(f"mBART translation failed: {response.status_code}")
