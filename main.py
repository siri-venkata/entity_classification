import time
ctime = time.time()
print('Started imports ')
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

print(f'Completed Loading modules in {time.time()-ctime} seconds')
ctime=time.time()

args = get_args()
wandb.init(
    # set the wandb project where this run will be logged
    project="entiity_classification",
    # track hyperparameters and run metadata
    config=vars(args),
    mode = args.wand_mode,
    name = args.wand_run_name,
    
)
print(f'Initialized logger  in {time.time()-ctime} seconds')
ctime=time.time()

print('Starting Run: ',wandb.run.name)
seed_all(args.seed)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
args.device = device


#Generate Label Maps
print('Generating Label Maps')
A,M,R,D,labels,label2id,id2label,num_labels = get_label_maps(args)
label_graph = load_label_graph(A,id2label, args)
args.num_labels = num_labels

print(f'Loaded data & labels in {time.time()-ctime} seconds')
ctime=time.time()

#Load Model and Tokenizer
print('Loading Model and Tokenizer')
tokenizer = load_tokenizer(args)
args.pad_token_id = tokenizer.pad_token_id
model = model_loader_dict[args.model_type](label_graph,args)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
untrainable_params = total_params - trainable_params

print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
print(f"Untrainable parameters: {untrainable_params}")

if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)

model.to(device)


print(f'Loaded Model        in {time.time()-ctime} seconds')
ctime=time.time()



def evaluate(model,eval_loader,eval_metric,eval_size,loss_fn,device,step_number,best_loss,args):
    evaluation_loss = Lossaggregator(batch_size=args.eval_batch_size)
    with torch.no_grad():
        eval_steps=0
        for index,batch in enumerate(eval_loader):
            input_ids = batch['input_ids'].to(device)#.reshape(args.batch_size,-1).to(device)
            attention_mask = batch['attention_mask'].to(device)#.reshape(args.batch_size,-1).to(device)
            labels = batch['labels'].to(device)#.reshape(args.batch_size,-1).to(device)

            outputs = model(input_ids=input_ids,attention_mask=attention_mask,**{i:batch[i] for i in batch if i not in {'input_ids','attention_mask','labels'}})
            outputs,loss = loss_fn(outputs,labels)

            eval_metric.add(outputs.detach().cpu(),labels.detach().cpu(),batch['meta_data'])
            evaluation_loss.add(loss.item())
            eval_steps+=1
            if eval_steps>args.eval_steps:
                break
            
    #print('Evaluation Loss: ',evaluation_loss.get())
    #print('Evaluation Metrics: ',eval_metric.get())
    val_loss = evaluation_loss.get()
    wandb.log({"Eval/loss": val_loss},step=step_number)
    wandb.log({"Eval/"+met:val for met,val in eval_metric.get(True).items()},step=step_number)
    if args.save_results:
        eval_metric.save_predictions(eval=True,name='last' if val_loss>=best_loss else 'best')
    evaluation_loss.reset()
    eval_metric.reset()
    return val_loss


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
          training_metric,
          eval_loader,
          eval_metric,
          train_size,
          eval_size,
          skip_eval,
          device,
          args):
    print('Training')
    wandb.watch(model)
    step_number = 0
    last_train_log =-1
    best_loss = 1_000_000_000
    while step_number==0 or step_number<args.train_steps:
        for i in range(args.num_train_epochs):
            print('Epoch: ',i)
            training_loss = Lossaggregator(batch_size=args.batch_size)
            
            for index,batch in tqdm(enumerate(training_loader),total = train_size):
                input_ids = batch['input_ids'].to(device)#.reshape(args.batch_size,-1).to(device)
                attention_mask = batch['attention_mask'].to(device)#.reshape(args.batch_size,-1).to(device)
                labels = batch['labels'].to(device)#.reshape(args.batch_size,-1).to(device)
                
                
                outputs = model(input_ids=input_ids,attention_mask=attention_mask,**{i:batch[i] for i in batch if i not in {'input_ids','attention_mask','labels'}})
                outputs,loss = loss_fn(outputs,labels)

                
                training_metric.add(outputs.detach().cpu(),labels.detach().cpu(),batch['meta_data'])
                training_loss.add(loss.item())

                if step_number%args.logging_steps==0 and step_number!=0:
                    wandb.log({'epoch':i},step=step_number)
                    wandb.log({'LR':scheduler.get_last_lr()[0]},step=step_number)
                    wandb.log({"Train/loss": training_loss.get()},step=step_number)
                    wandb.log({"Train/"+met:val for met,val in training_metric.get().items()},step=step_number)
                    last_train_log = step_number

                    

                loss.backward()
                if args.max_grad_norm>0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if step_number%args.gradient_accumulation_steps==0 and step_number!=0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if (not skip_eval) and (step_number%args.eval_num_steps==0) and step_number!=0:
                    if last_train_log!=step_number:
                        wandb.log({'epoch':i},step=step_number)
                        wandb.log({'LR':scheduler.get_last_lr()[0]},step=step_number)
                        wandb.log({"Train/loss": training_loss.get()},step=step_number)
                        wandb.log({"Train/"+met:val for met,val in training_metric.get(True).items()},step=step_number)
                        last_train_log=step_number
                    model.eval()
                    val_loss = evaluate(model,eval_loader,eval_metric,eval_size,loss_fn,device,step_number,best_loss,args)
                    if val_loss<best_loss:
                        val_loss = best_loss
                        if args.save_model:
                            save_checkpoint(model,optimizer,scheduler,True,'best',args)
                    model.train()
                step_number+=1
                if step_number>=args.train_steps:
                    print('Exiting Current Epoch')
                    break
            if step_number>=args.train_steps:
                print('Exiting epoch loop')
                break

            training_loss.reset()
            if args.save_results:
                training_metric.save_predictions(False)
            training_metric.reset()
            if args.save_model:
                save_checkpoint(model,optimizer,scheduler,False,'last',args)

        if step_number>=args.train_steps or args.train_steps==1_000_000_000:
            print('Exiting Training loop')
            break


if __name__=="__main__":
    if args.do_train or args.do_eval:
        training_loader,train_size,eval_loader,eval_size = get_data_loaders(A,label2id,id2label,tokenizer,args)
        M= torch.from_numpy(A).float()
        training_metric = Metricsaggregator(M,id2label,'train',args)
        eval_metric = Metricsaggregator(M,id2label,'eval',args)
        optimizer,scheduler = get_optimizer_and_scheduler(model,train_size,args)
        loss_fn = loss_functions[args.loss_type]
        M = M.to(device)
        R = torch.from_numpy(R).float().to(device)
        D = torch.from_numpy(D).float().to(device)
        loss_fn = add_variable_to_scope(M=M,R=R,D=D)(loss_fn)
    if args.do_train and args.do_eval:
        train(model,optimizer,scheduler,loss_fn,training_loader,training_metric,eval_loader,eval_metric,train_size,eval_size,False,device,args)
    elif args.do_train:
        train(model,optimizer,scheduler,loss_fn,training_loader,training_metric,eval_loader,eval_metric,train_size,eval_size,True,device,args)
    elif args.do_eval:
        evaluate(model,eval_loader,eval_metric,eval_size,loss_fn,device,args)
    elif args.do_predict:
        predict_loader,predict_set = predict_data_loader()
        predictions = predict(model,predict_loader,predict_set,device,args)
        with open('/'.join(args.predict_file.split('/')[:-1]),'/predictions.json','w') as f:
            json.dump(predictions,f)
    
