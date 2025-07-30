import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from backend.llm_model import TinyTransformer
import os

# Load local text
with open("data/text_corpus.txt", "r", encoding="utf-8") as f:
    texts = f.readlines()

# Tokenize
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
encoded = [tokenizer.encode(t.strip(), truncation=True, max_length=128) for t in texts]
padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(e) for e in encoded], batch_first=True)

# Model setup
model = TinyTransformer(vocab_size=tokenizer.vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

# Train
for epoch in range(50):
    for batch in DataLoader(padded, batch_size=2, shuffle=True):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# Save
os.makedirs("models/saved_weights", exist_ok=True)
torch.save(model.state_dict(), "models/saved_weights/tiny_transformer.pth")
