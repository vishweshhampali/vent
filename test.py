
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the tokenizer
# tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')

# Add special tokens
#special_tokens_dict = {'additional_special_tokens': ['<speaker0>', '<speaker1>']}
#tokenizer.add_special_tokens(special_tokens_dict)
#tokenizer.pad_token = tokenizer.eos_token  # Ensure a valid pad token is set

max_length = 512

# Load the DialoGPT model
#model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')
#model.resize_token_embeddings(len(tokenizer))
#model.to(device)  # Move model to GPU


# Load the model and tokenizer for inference
model = AutoModelForCausalLM.from_pretrained('./dialogpt-finetuned').to(device)
tokenizer = AutoTokenizer.from_pretrained('./dialogpt-finetuned')
model.resize_token_embeddings(len(tokenizer))

special_tokens_dict = {'additional_special_tokens': ['<speaker0>', '<speaker1>']}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token  # Ensure a valid pad token is set


# Response generation function
def generate_response(conversation_history, max_length=512):
        input_ids = tokenizer.encode(conversation_history + tokenizer.eos_token, return_tensors='pt').to(device)
        output = model.generate(
                input_ids,
                max_length=max_length,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=1,
                )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        return response

# Example conversation
conversation = "I had a tough day at work."
response = generate_response(conversation)
print(conversation)
print(response)

conversation = "I got punished at school for not doing my homework."
response = generate_response(conversation)
print(conversation)
print(response)

conversation = "Hello, I really dont feel good. My girlfriend broke up with me."
response = generate_response(conversation)
print(conversation)
print(response)

