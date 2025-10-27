#!/bin/bash
# Test HF Inference API endpoints

echo "Testing Hugging Face Inference API..."

# Get API key from .env
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

echo ""
echo "1. Testing mBART translation:"
curl -X POST \
  "https://api-inference.huggingface.co/models/facebook/mbart-large-50-many-to-many-mmt" \
  -H "Authorization: Bearer $HUGGINGFACE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "en_XX Hello world"}'

echo ""
echo "2. Testing BART summarization:"
curl -X POST \
  "https://api-inference.huggingface.co/models/facebook/bart-large-cnn" \
  -H "Authorization: Bearer $HUGGINGFACE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"inputs": "The tower is 324 metres tall, about the same height as an 81-storey building."}'

