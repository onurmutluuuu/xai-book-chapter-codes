# Required libraries: pip install transformers torch
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt

print("\n--- Transformer Attention Visualization ---")

# Load the model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_attentions=True)

# Sample sentence
sentence = "The cat sat on the mat"
inputs = tokenizer(sentence, return_tensors='pt')
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]) # Fix: Added [0] for token conversion

# Execute the model and retrieve attention weights
with torch.no_grad():
    outputs = model(**inputs)
attentions = outputs.attentions  # This is a tuple containing each of the 12 layers

# Technical Correction: 'attentions' is a list/tuple, so .detach() cannot be applied directly.
# Here we select the first layer (0), the first batch [0], and the first attention head [0].
attention_layer_0_head_0 = attentions[0][0, 0].detach().numpy() 

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))
im = ax.imshow(attention_layer_0_head_0, cmap='viridis')

ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, rotation=45) # 45-degree rotation for better readability
ax.set_yticklabels(tokens)
ax.set_title("BERT Attention Map (Layer 0, Head 0)")
fig.colorbar(im, ax=ax)
plt.show()

print("Description: The matrix illustrates the 'attention' weight each word (vertical axis) assigns to other words (horizontal axis).")
print("For instance, the word 'sat' is likely to exhibit strong attention towards 'cat' and 'mat' to capture contextual relations.")

