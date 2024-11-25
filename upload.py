
from transformers import AutoModelForCausalLM, AutoTokenizer

from huggingface_hub import login
login()

from huggingface_hub import HfApi


model = AutoModelForCausalLM.from_pretrained('./dialogpt-finetuned')
tokenizer = AutoTokenizer.from_pretrained('./dialogpt-finetuned')

api = HfApi()
model_repo_name = "vishweshhampali/dialogpt-finetuned"  # Format of Input  <Profile Name > / <Model Repo Name> 

#Create Repo in Hugging Face
#api.create_repo(repo_id=model_repo_name)

model_path = "dialogpt-finetuned"
#Upload Model folder from Local to HuggingFace 
api.upload_folder(
            folder_path=model_path,
                repo_id=model_repo_name
                )

# Publish Model Tokenizer on Hugging Face
tokenizer.push_to_hub(model_repo_name)

