import os
import json
import torch
from utils.read_data import create_data_loader,get_label_maps
from utils.load_model import load_model
from transformers import TrainingArguments, Trainer



model_name='bert-base-multilingual-cased'
model_path =''
batch_size = 8
max_input_len = 512
metric_name = 'f1'




#Load Label Maps
num_labels,id2label,label2id = get_label_maps()
model,tokenizer = load_model(model_name,model_path,num_labels,id2label,label2id)



#Get data
data = get_data(lang='en')

split = int(len(data)*0.8)
train_dataset = EntityDataset(data[:split], tokenizer, label2id, max_len=max_input_len, float_labels=True,flatten_label=True)
eval_dataset = EntityDataset(data[split:], tokenizer, label2id, max_len=max_input_len, float_labels=True,flatten_label=True)

# training_loader = create_data_loader(data[:split], tokenizer, max_len=max_input_len, batch_size=batch_size, flatten_label=True)
# test_loader = create_data_loader(data[split:], tokenizer, max_len=max_input_len, batch_size=batch_size, flatten_label=True)


args = TrainingArguments(
    f"mbert_flat",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model=metric_name,
)

trainer = Trainer(
    model,
    args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


trainer.train()


trainer.evaluate()