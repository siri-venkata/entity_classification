import os
import json
import torch
from utils.read_data import create_data_loader,get_label_maps,get_data,EntityDataset
from utils.load_model import load_model,load_tokenizer
from utils.metrics import Metricsaggregator
from transformers import TrainingArguments, Trainer
from utils.losses import loss_functions,Lossaggregator
from tqdm import tqdm



model_name='bert-base-multilingual-cased'
model_path =''
batch_size = 8
max_input_len = 512
metric_name = 'f1'
num_train_epochs = 5
loss_type='mc'



#Generate Label Maps
print('Generating Label Maps')
A,M,labels,id2label,num_labels = get_label_maps()

#Load Model and Tokenizer
print('Loading Model and Tokenizer')
model = load_model(model_name,model_path,num_labels,id2label)
tokenizer = load_tokenizer(model_name,model_path)


#Get data
print('Loading Data')
data = get_data(lang='en')

split = int(len(data)*0.8)
train_dataset = EntityDataset(data[:split], A, id2label, tokenizer,  max_len=max_input_len, flatten_label=True)
eval_dataset = EntityDataset(data[split:], A, id2label, tokenizer,  max_len=max_input_len, flatten_label=True)

training_loader = create_data_loader(train_dataset,batch_size=batch_size)
eval_loader = create_data_loader(eval_dataset, batch_size=batch_size)


# args = TrainingArguments(
#     f"mbert_flat",
#     evaluation_strategy = "epoch",
#     save_strategy = "epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=batch_size,
#     per_device_eval_batch_size=batch_size,
#     num_train_epochs=num_train_epochs,
#     weight_decay=0.01,
#     load_best_model_at_end=True,
#     metric_for_best_model=metric_name,
# )

# trainer = Trainer(
#     model,
#     args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     tokenizer=tokenizer,
#     compute_metrics=compute_metrics
# )

# print('Training')

# trainer.train()

# print('Evaluating')
# trainer.evaluate()





device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
model.to(device)
M = torch.from_numpy(M).float().to(device)

loss_fn = loss_functions[loss_type]
for i in range(num_train_epochs):
    print('Epoch: ',i)
    training_loss = Lossaggregator(batch_size=batch_size)
    training_metric = Metricsaggregator()

    for index,batch in tqdm(enumerate(training_loader),total = len(train_dataset)//batch_size):
        input_ids = batch['input_ids'].reshape(batch_size,-1).to(device)
        labels = batch['labels'].reshape(batch_size,-1).to(device)

        outputs = model(input_ids).logits
        outputs,loss = loss_fn(outputs,labels,M)

        training_metric.add(outputs.detach().cpu(),labels.detach().cpu())
        training_loss.add(loss.item())

        if index%100==0:
            print('Loss: ',training_loss.get())
            print('Metrics: ',training_metric.get())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print('Evaluating')
    evaluation_loss = Lossaggregator(batch_size=batch_size)
    evaluation_metric = Metricsaggregator()
    with torch.no_grad():
        for index,batch in tqdm(enumerate(training_loader),total = len(eval_dataset)//batch_size):
            input_ids = batch['input_ids'].reshape(batch_size,-1).to(device)
            labels = batch['labels'].reshape(batch_size,-1).to(device)

            outputs = model(input_ids).logits
            outputs,loss = loss_fn(outputs,labels,M)

            evaluation_metric.add(outputs.detach().cpu(),labels.detach().cpu())
            evaluation_loss.add(loss.item())
            
    print('Evaluation Loss: ',evaluation_loss.get())
    print('Evaluation Metrics: ',evaluation_metric.get())
    
    training_loss.reset()
    evaluation_loss.reset()
    training_metric.reset()
    evaluation_metric.reset()

            