from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("loading model and tokenizer...")
model_path = "./dialogpt-finetuned"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)


special_tokens_dict = {'additional_special_tokens':['<speaker0>','<speaker1>']}
tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token':'[PAD]'})
    model.resize_token_embeddings(len(tokenizer))


def generate_response(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = tokenizer.eos_token_id,
            do_sample = True,
            top_k = 50,
            top_p = 0.95,
            temperature=0.7,
            num_return_sequences=1,
            )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    response = response[len(prompt):].strip()
    return response


def chat():
    
    print("Start chatting wih the bot (type 'exit' to stop)...")
    while True:
        user_input = input("User: ")
        
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        response = generate_response(user_input)

        print(f"Chatbot: {response}")

if __name__=='__main__':
    chat()
