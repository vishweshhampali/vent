from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from transformers import TrainingArguments, Trainer

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")



model = AutoModelForCausalLM.from_pretrained('./dialogpt-finetuned').to(device)
tokenizer = AutoTokenizer.from_pretrained('./dialogpt-finetuned')

# Add special tokens
special_tokens_dict = {'additional_special_tokens': ['<speaker0>', '<speaker1>']}
tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.pad_token = tokenizer.eos_token  # Ensure a valid pad token is set

def  generate_response(conversation_history, max_length=512):
    history = ''
    for speaker_idx, utterance in conversation_history:
        #speaker_idx = speaker_idx % 2
        speaker_token = f'<speaker{speaker_idx}>'
        history += f'{speaker_token}: {utterance} <|endoftext|>'

    bot_speaker_token = '<speaker1>'
    prompt = history + f'{bot_speaker_token}:'

    print("Prompt:", prompt)

    inputs = tokenizer(prompt, return_tensors='pt', padding=True).to(device)
    input_ids = inputs['input_ids']

    #input_ids = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors='pt').to(device)
    print('Input Ids:', input_ids)

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

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    
    print('Raw Generated:', generated)

    prompt_length = len(tokenizer.decode(input_ids[0], skip_special_tokens=True))
    response = generated[prompt_length:].strip()
    end_tokens = ['<|endoftext|>', '<speaker0>','<speaker1>']

    for token in end_tokens:
        if token in response:
            response = response.split(token)[0].strip()

    return response

conversation_history = []
print("Chatbot is ready! Type 'exit' tp end the conversation.")

while True:
    user_input = input("You:")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    conversation_history.append((0, user_input))
    print("Conversation History:",conversation_history)
    bot_response = generate_response(conversation_history)

    print("Bot:", bot_response)

    conversation_history.append((1, bot_response))
