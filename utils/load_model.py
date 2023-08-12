import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_name, model_path='',num_labels=10,id2label=None):
    if model_path=='':
        model_path = model_name
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                            problem_type="multi_label_classification", 
                                                           num_labels=num_labels,
                                                           id2label=id2label)
    return model


def load_tokenizer(model_name, model_path=''):
    if model_path=='':
        model_path = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer


def dummy_config_model():
    model_name='bert-base-multilingual-cased'
    model_path =''
    labels=['Cat','Dog','Horse','Cow','Sheep','Goat','Pig','Chicken','Duck','Goose']
    num_labels = len(labels)
    id2label = {i:label for i,label in enumerate(labels)}
    label2id = {label:i for i,label in enumerate(labels)}
    model,tokenizer = load_model(model_name,model_path,num_labels,id2label,label2id)
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
    print('*************')

    ouptut = test_model(model,input,text_labels)


