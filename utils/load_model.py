import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_model(model_name, model_path='',num_labels=10,id2label={},label2id={}):
    if model_path=='':
        model_path = model_name
    model = AutoModelForSequenceClassification.from_pretrained(model_path,
                                                            problem_type="multi_label_classification", 
                                                           num_labels=len(labels),
                                                           id2label=id2label,
                                                           label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model,tokenizer
