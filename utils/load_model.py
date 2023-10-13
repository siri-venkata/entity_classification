import os
import torch
import shutil
from utils.utils import AttrDict
from transformers import AutoModel,AutoModelForSequenceClassification, AutoTokenizer

def load_model(args):
    if args.model_path=='':
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name,
                                                            problem_type="multi_label_classification", 
                                                           num_labels=args.num_labels)
        return model
    else:
        model = AutoModelForSequenceClassification()
        # .from_pretrained(model_path,
        #                                                     problem_type="multi_label_classification", 
        #                                                    num_labels=num_labels,
        #                                                    id2label=id2label)
        checkpoint = torch.load(args.model_path+'/model.pt')
        model.load_state_dict(checkpoint)
        return model

def load_base_model(args):
    model = AutoModel.from_pretrained(args.model_name if args.model_path=='' else args.model_path)
    return model

def save_checkpoint(model,optimizer,scheduler, is_best, step_number,args):
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
        os.makedirs(args.save_model_path+'/best')
    mpath = args.save_model_path+'/model_'+str(step_number)+'.pt'
    opath = args.save_model_path+'/optimizer_'+str(step_number)+'.pt'
    spath = args.save_model_path+'/scheduler_'+str(step_number)+'.pt'
    torch.save(model.state_dict(),mpath)
    torch.save(optimizer.state_dict(),opath)
    torch.save(scheduler.state_dict(),spath)

    if is_best:
        shutil.copyfile(mpath,'/'.join(mpath.split('/')[:-1])+'/best/model_'+str(step_number)+'.pt')
        shutil.copyfile(opath,'/'.join(opath.split('/')[:-1])+'/best/optimizer_'+str(step_number)+'.pt')
        shutil.copyfile(spath,'/'.join(spath.split('/')[:-1])+'/best/scheduler_'+str(step_number)+'.pt')
        os.remove(mpath)
        os.remove(opath)
        os.remove(spath)

def load_tokenizer(args,**kwargs):
    if args.model_path=='':
        model_path = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast =kwargs["use_fast"] if "use_fast" in kwargs else True)
    return tokenizer


def dummy_config_model():
    model_name='bert-base-multilingual-cased'
    model_path =''
    labels=['Cat','Dog','Horse','Cow','Sheep','Goat','Pig','Chicken','Duck','Goose']
    num_labels = len(labels)
    id2label = {i:label for i,label in enumerate(labels)}
    label2id = {label:i for i,label in enumerate(labels)}
    model = load_model(num_labels,id2label,AttrDict({"model_name":model_name,"model_path":model_path}))
    tokenizer = load_tokenizer(AttrDict({"model_name":model_name,"model_path":model_path}))
    return model,tokenizer,labels

def test_model(model,input,labels=None):
    output = model(**input) if labels is None else model(**input,labels=labels)
    for i in input.keys():
        print(i)
        print(input[i].shape)

    print(output.keys())
    for i in output.keys():
        print(i)
        print(output[i].shape)
    
    return output



if __name__=="__main__":
    model,tokenizer,labels = dummy_config_model()
    #Create a batch of 2 sentences
    texts = ["Hello, this one sentence!","This is another sentence"]
    text_labels = torch.Tensor([[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0]])

    input = tokenizer(texts, return_tensors="pt",padding=True)
    output = test_model(model,input)
    print(output.logits.shape)
    print('*************')

    ouptut = test_model(model,input,text_labels)


