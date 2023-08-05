import os
import json
from torch.utils.data import Dataset, DataLoader


articles_path ='data/articles/enwiki.json'
id_path = 'data/articles/lang_ids.json'
labels_path = 'data/small_inverted_index.json'
type_path = 'data/small_type_counts.json'

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def get_label_maps():
    labels = read_json(type_path)
    def flatten_dict(d):
        if type(d)==str or type(d)==int:
            return [str(d)]
        res = [[i]+flatten_dict(d[i]) for i in d]
        return [j for i in res for j in i]
    labels = flatten_dict(labels)
    labels = [i.replace('.txt','') for i in labels if not i.isnumeric()]
    labels = list(set(labels))
    id2label = {i:label for i,label in enumerate(labels)}
    label2id = {label:i for i,label in enumerate(labels)}
    return len(labels),id2label,label2id


def get_data(lang='en'):
    #Load lang_title_articles
    articles_path=articles_path.replace('en',lang)
    articles = read_json(articles_path)

    #Load lang_id lookup
    
    ids = read_json(id_path)
    ids = ids[lang+'wiki']

    #Load id label lookup
    
    labels = read_json(labels_path)

    data = []
    for title,content in articles.items():
        id = ids[title]
        label = labels[id]
        data.append([id,lang,title,content,label])
    return data

def flatten_label(label): return set(list([j for i in label for j in i]))


#Pytorch Dataset
class EntityDataset(Dataset):
    def __init__(self, data, tokenizer, label2id, max_len=512, float_labels=True,flatten_label=True):
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        self.float_labels = float_labels
        self.flatten_label = flatten_label


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        id,lang,title,content,label =self.data[item]
        if self.flatten_label:
            label = flatten_label(label)
        if self.float_labels:
            label = [self.label2id[i] for i in label]
            labels = torch.zeros(len(self.label2id))
            labels[label] = 1
            label = labels

        encoding = self.tokenizer.encode_plus(
            [title+
            content],
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_tensors='pt',
        )
        return {"input":encoding[0],"label":label,"id":id,"lang":lang}


def create_data_loader(ds, batch_size=8):
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )

    
if __name__=="__main__":
    num_labels,id2label,label2id = get_label_maps()
    print(num_labels)
    import pdb;pdb.set_trace()