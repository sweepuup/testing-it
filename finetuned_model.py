import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

checkpoint = 'gpt2'
# model_path = 'C:/Users/dinis/.cache/huggingface/finetuned/gpt2_nsfw_finetune.pt'
model_path = 'C:YOUR/FILE/PATH'
model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model.load_state_dict(torch.load(model_path))
model.eval()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
pipeline_device = 0 if torch.cuda.is_available() else -1

base_generator = pipeline('text-generation', model=model, tokenizer=tokenizer, max_length=69,
                          torch_dtype=torch.bfloat16, trust_remote_code=True,
                          temperature=1.0, top_k=50, top_p=0.8, do_sample=True)
print('PRE-FINETUNED TEXT-GEN: ', base_generator('Hey... Do you want to'))