from transformers import pipeline
from transformers import AutoModel, AutoTokenizer

prober = pipeline('fill-mask', model="bert-base-cased", top_k=50, framework="pt", batch_size=32)
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)
vocab_size = tokenizer.vocab_size
ids2tokens = tokenizer.ids_to_tokens
vocab = tokenizer.vocab  # tokens2ids
mask = tokenizer.mask_token
mask_id = vocab[mask]
print()