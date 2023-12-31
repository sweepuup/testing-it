import torch
from torch.utils.checkpoint import checkpoint as grad_check
from torch.utils.data import DataLoader
import evaluate
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, AdamW, get_scheduler
from tqdm.auto import tqdm
from datasets import load_dataset

# SAVE_PATH = 'C:/Users/dinis/.cache/huggingface/finetuned/gpt2_nsfw_finetune.pt'
SAVE_PATH = 'C:YOUR/FILE/PATH'
checkpoint = 'gpt2'
base_model = AutoModelForCausalLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
raw_dataset = load_dataset('mickume/alt_nsfw')
split = raw_dataset['train'].select(range(round(len(raw_dataset['train']) / 1000))).train_test_split(test_size=0.2)
train_dataset = split['train']
eval_dataset = split['test']
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
pipeline_device = 0 if torch.cuda.is_available() else -1


def tokenize_function(example):
    return tokenizer(example['text'], truncation=True, padding=True)


### DATA PREPROCESSING ###
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)
tokenized_train_dataset = tokenized_train_dataset.remove_columns('text')
tokenized_eval_dataset = tokenized_eval_dataset.remove_columns('text')
tokenized_train_dataset.set_format('torch')
tokenized_eval_dataset.set_format('torch')
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_dataloader = DataLoader(
    tokenized_train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_eval_dataset, batch_size=4, collate_fn=data_collator
)

print('Preparing batch in train_dataloader...')
optimizer = AdamW(base_model.parameters(), lr=5e-5)
EPOCHS = 2
num_training_steps = EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)


print('Putting base_model to device...')
torch.cuda.empty_cache()
### TRAINING LOOP ###
base_model.to(device)
progress_bar = tqdm(range(num_training_steps))
base_model.train()
print('Starting training loop...')
for epoch in range(EPOCHS):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch['labels'] = batch['input_ids'].clone()
        outputs = base_model(**batch)
        loss = outputs.loss
        if loss is not None:
            loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        print('EPOCH: ', epoch)
        print('loss: ', loss)
# metric = evaluate.load('mickume/alt_nsfw')
# base_model.eval()
# for batch in eval_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = base_model(**batch)
#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     metric.add_batch(predictions=predictions)
# metric.compute()


torch.save(base_model.state_dict(), SAVE_PATH)
