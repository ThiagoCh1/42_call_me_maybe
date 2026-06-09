"""Temporary test file to explore the llm_sdk methods.

Run with: uv run python tests/test_sdk.py
"""

from llm_sdk import Small_LLM_Model

print("Loading model... (this may take a while the first time)")
model = Small_LLM_Model()
print("Model loaded.\n")

# --- encode ------------------------------------------------------------------
# Converts text into a list of token IDs
text = "What is the sum of 2 and 3?"
encoded = model.encode(text)
print(f"Text: {text}")
print(f"Encoded (input_ids tensor): {encoded}")
print(f"Shape: {encoded.shape}\n")

# --- decode ------------------------------------------------------------------
# Converts token IDs back into text
ids = encoded[0].tolist()
decoded = model.decode(ids)
print(f"IDs: {ids}")
print(f"Decoded back to text: {decoded}\n")

# --- get_logits_from_input_ids -----------------------------------------------
# Given a list of token IDs, returns the logits for the next token
# (one score per token in the vocabulary)
logits = model.get_logits_from_input_ids(ids)
print(f"Number of logits (= vocabulary size): {len(logits)}")
print(f"First 10 logits: {logits[:10]}")

# The token with the highest logit is the most likely next token
best_token_id = logits.index(max(logits))
print(f"Most likely next token ID: {best_token_id}")
print(f"Most likely next token text: '{model.decode([best_token_id])}'\n")

# --- get_path_to_vocab_file --------------------------------------------------
# Returns the path to the vocabulary JSON file
# This maps token IDs to their string representations
vocab_path = model.get_path_to_vocab_file()
print(f"Vocab file path: {vocab_path}\n")

# --- peek at the vocabulary --------------------------------------------------
import json
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

print(f"Vocabulary size: {len(vocab)}")
print(f"Sample entries (first 5): {list(vocab.items())[:5]}")
print(f"Token ID for best token: {vocab.get(model.decode([best_token_id]))}")
