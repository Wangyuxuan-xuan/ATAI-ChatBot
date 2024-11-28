import torch
from transformers import pipeline

# Model identifier for the instruct version of Llama
model_id = "meta-llama/Llama-3.2-1B-Instruct"
access_token = "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"

# Setting up the pipeline for text generation with the instruct model
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",  # Automatically chooses the right device (GPU/CPU)
    token =access_token  # Pass your access token for gated model access
)

# Example messages for the conversation
messages = [
    {"role": "system", "content": "You are a specialized movie chatbot to answer user queries in short sentences."},
    {"role": "user", "content": "When was the Godfather released?"}
]

# Generate the output
outputs = pipe(
    messages,  # Conversation messages
    max_new_tokens=256  # Maximum number of tokens to generate
)

# Print the generated response from the assistant
print(outputs[0]["generated_text"])
