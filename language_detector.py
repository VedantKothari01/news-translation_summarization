from langdetect import detect, LangDetectException

def detect_language(text, threshold=0.5):
    try:
        if not text or len(text.strip()) < 3:
            return 'en'
        detected = detect(text[:200])
        return detected
    except LangDetectException:
        return 'en'
    except Exception:
        return 'en'
