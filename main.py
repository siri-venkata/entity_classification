
import os
import json
import wandb


import torch
from utils.arguments import get_args
from utils.read_data import create_data_loader,get_label_maps,load_label_graph,get_data_loaders,predict_data_loader
from utils.load_model import load_model,load_tokenizer,save_checkpoint


from models import model_loader_dict
from utils.metrics import Metricsaggregator
from utils.utils import seed_all
from utils.optimizers import get_optimizer_and_scheduler
from utils.losses import loss_functions,Lossaggregator,add_variable_to_scope
from tqdm import tqdm



args = get_args()
wandb.init(
    # set the wandb project where this run will be logged
    project="entiity_classification",
    # track hyperparameters and run metadata
    config=vars(args),
    mode = "disabled"
)

print('Starting Run: ',wandb.run.name)
seed_all(args.seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
args.device = device


#Generate Label Maps
print('Generating Label Maps')
A,M,R,D,labels,label2id,id2label,num_labels = get_label_maps(args)
label_graph = load_label_graph(A,id2label, args)
args.num_labels = num_labels

#Load Model and Tokenizer
print('Loading Model and Tokenizer')
tokenizer = load_tokenizer(args)
args.pad_token_id = tokenizer.pad_token_id
model = model_loader_dict[args.model_type](label_graph,args)


model.to(device)




def evaluate(model,eval_loader,eval_size,loss_fn,device,args):
    print('Evaluating')
    evaluation_loss = Lossaggregator(batch_size=args.eval_batch_size)
    evaluation_metric = Metricsaggregator(args = args)
    with torch.no_grad():
        eval_steps=0
        for index,batch in tqdm(enumerate(eval_loader),total = eval_size//args.eval_batch_size):
            input_ids = batch['input_ids'].to(device)#.reshape(args.batch_size,-1).to(device)
            attention_mask = batch['attention_mask'].to(device)#.reshape(args.batch_size,-1).to(device)
            labels = batch['labels'].to(device)#.reshape(args.batch_size,-1).to(device)

            outputs = model(input_ids=input_ids,attention_mask=attention_mask,**{i:batch[i] for i in batch if i not in {'input_ids','attention_mask','labels'}})
            outputs,loss = loss_fn(outputs,labels)

            evaluation_metric.add(outputs.detach().cpu(),labels.detach().cpu())
            evaluation_loss.add(loss.item())
            eval_steps+=1
            if eval_steps>args.eval_steps:
                break
            
    print('Evaluation Loss: ',evaluation_loss.get())
    print('Evaluation Metrics: ',evaluation_metric.get())
    evaluation_metric.save_predictions(eval=True)
    wandb.log({"Eval loss": evaluation_loss.get()})
    wandb.log({"Eval "+met:val for met,val in evaluation_metric.get().items()})



def predict(model,
            predict_loader,
            predict_set,
            loss_fn,
            device,
            args):
    print('Predicting')
    predictions = []
    model.eval()
    with torch.no_grad():
        for index,batch in tqdm(enumerate(predict_loader),total = len(predict_set)//args.batch_size):
            input_ids = batch['input_ids'].reshape(args.batch_size,-1).to(device)
            attention_mask = batch['attention_mask'].reshape(args.batch_size,-1).to(device)

            outputs = model(input_ids=input_ids,attention_mask=attention_mask,**{i:batch[i] for i in batch if i not in {'input_ids','attention_mask','labels'}})
            outputs,loss = loss_fn(outputs,labels)
            predictions.append(outputs.detach().cpu().numpy())
        return predictions






def train(model,
          optimizer,
          scheduler,
          loss_fn,
          training_loader,
          eval_loader,
          train_size,
          eval_size,
          skip_eval,
          device,
          args):
    print('Training')
    step_number = 0
    best_loss = 1_000_000_000
    while step_number==0 or step_number<args.train_steps:
        for i in range(args.num_train_epochs):
            print('Epoch: ',i)
            training_loss = Lossaggregator(batch_size=args.batch_size)
            training_metric = Metricsaggregator(args = args)
            for index,batch in tqdm(enumerate(training_loader),total = train_size//args.batch_size):
                input_ids = batch['input_ids'].to(device)#.reshape(args.batch_size,-1).to(device)
                attention_mask = batch['attention_mask'].to(device)#.reshape(args.batch_size,-1).to(device)
                labels = batch['labels'].to(device)#.reshape(args.batch_size,-1).to(device)
                
                
                outputs = model(input_ids=input_ids,attention_mask=attention_mask,**{i:batch[i] for i in batch if i not in {'input_ids','attention_mask','labels'}})
                outputs,loss = loss_fn(outputs,labels)

                
                training_metric.add(outputs.detach().cpu(),labels.detach().cpu())
                training_loss.add(loss.item())

                if step_number%args.logging_steps==0:
                    # print('Loss: ',training_loss.get())
                    # print('Metrics: ',training_metric.get())
                    wandb.log({"Training loss": training_loss.get()})
                    wandb.log({"Training "+met:val for met,val in training_metric.get().items()})

                if step_number%args.saving_steps==0 and step_number!=0 and training_loss.get()<best_loss:
                    best_loss = training_loss.get()
                    save_checkpoint(model,optimizer,scheduler,True,step_number,args)
                    training_metric.save_predictions(False)
                    

                loss.backward()
                if args.max_grad_norm>0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if step_number%args.gradient_accumulation_steps==0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if (not skip_eval) and (step_number%args.eval_num_steps==0):
                    model.eval()
                    evaluate(model,eval_loader,eval_size,loss_fn,device,args)
                    model.train()

                step_number+=1
                if step_number>=args.train_steps:
                    print('Exiting epoch loop')
                    break
            training_loss.reset()
            training_metric.reset()
            save_checkpoint(model,optimizer,scheduler,False,step_number,args)
            if step_number>=args.train_steps:
                print('Exiting Training loop')
                break


if __name__=="__main__":
    if args.do_train or args.do_eval:
        training_loader,train_size,eval_loader,eval_size = get_data_loaders(A,label2id,id2label,tokenizer,args)
        optimizer,scheduler = get_optimizer_and_scheduler(model,train_size,args)
        loss_fn = loss_functions[args.loss_type]
        M = torch.from_numpy(M).float().to(device)
        R = torch.from_numpy(R).float().to(device)
        D = torch.from_numpy(D).float().to(device)
        loss_fn = add_variable_to_scope(M=M,R=R,D=D)(loss_fn)
    if args.do_train and args.do_eval:
        train(model,optimizer,scheduler,loss_fn,training_loader,eval_loader,train_size,eval_size,False,device,args)
    elif args.do_train:
        train(model,optimizer,scheduler,loss_fn,training_loader,eval_loader,train_size,eval_size,True,device,args)
    elif args.do_eval:
        evaluate(model,eval_loader,eval_size,loss_fn,device,args)
    elif args.do_predict:
        predict_loader,predict_set = predict_data_loader()
        predictions = predict(model,predict_loader,predict_set,device,args)
        with open('/'.join(args.predict_file.split('/')[:-1]),'/predictions.json','w') as f:
            json.dump(predictions,f)
    