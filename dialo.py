from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the DailyDialog dataset
dataset = load_dataset('facebook/empathetic_dialogues', trust_remote_code=True)

# Define a function to clean the text
def clean_text(example):
    example['prompt'] = example['prompt'].replace('_comma_', ',')
    example['utterance'] = example['utterance'].replace('_comma_', ',')
    return example

# Apply the cleaning function
dataset = dataset.map(clean_text, batched=False)

# Split the dataset
train_dataset = dataset['train']
val_dataset = dataset['validation']
test_dataset = dataset['test']

# Preprocess data to create formatted conversations
def preprocess_data(dataset_split):
    conversations = {}
    for entry in dataset_split:
        conv_id = entry['conv_id']
        if conv_id not in conversations:
            conversations[conv_id] = []
        conversations[conv_id].append((entry['utterance_idx'], entry['speaker_idx'], entry['utterance']))

    # Sort and format each conversation
    for conv_id in conversations:
        conversations[conv_id] = sorted(conversations[conv_id], key=lambda x: x[0])
        
    formatted_data = []
    for conv in conversations.values():
        history = ''
        for idx, (utterance_idx, speaker_idx, utterance) in enumerate(conv):
            speaker = f'<speaker{speaker_idx}>'
            history += f'{speaker}: {utterance} <|endoftext|> '
            if idx >= 1:
                formatted_data.append(history.strip())
    return formatted_data

train_data = preprocess_data(train_dataset)
val_data = preprocess_data(val_dataset)
test_data = preprocess_data(test_dataset)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-medium')

# Add special tokens
special_tokens_dict = {'additional_special_tokens': ['<speaker0>', '<speaker1>']}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token  # Ensure a valid pad token is set


max_length = 512

# Tokenization function
def tokenize_data(data):
    return tokenizer(
        data,
        truncation=True,
        max_length=512,
        padding='max_length',
        return_tensors='pt'
    )

train_encodings = tokenize_data(train_data)
val_encodings = tokenize_data(val_data)

# Custom Dataset class
class DialogueDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings['input_ids']
        self.attention_masks = encodings['attention_mask']
        self.labels = self.input_ids.clone()

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

train_dataset = DialogueDataset(train_encodings)
val_dataset = DialogueDataset(val_encodings)

# Load the DialoGPT model
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small')
model.resize_token_embeddings(len(tokenizer))
model.to(device)  # Move model to GPU

# Training arguments
training_args = TrainingArguments(
    output_dir='./dialogpt-medium-finetuned',
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    fp16=True,  # Enable mixed precision
    save_steps=5000,
    save_total_limit=1,
    eval_steps=5000,
    logging_steps=500,
    learning_rate=5e-5,
    evaluation_strategy='steps',
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    report_to='none'
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

# Train the model
trainer.train()

# Save the model and tokenizer
model.save_pretrained('./dialogpt-finetuned')
tokenizer.save_pretrained('./dialogpt-finetuned')

# Load the model and tokenizer for inference
model = AutoModelForCausalLM.from_pretrained('./dialogpt-finetuned').to(device)
tokenizer = AutoTokenizer.from_pretrained('./dialogpt-finetuned')

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
print(response)

