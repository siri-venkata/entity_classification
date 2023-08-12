import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import functools
import itertools
from tqdm import tqdm


articles_path ='data/articles/enwiki.json'
id_path = 'data/articles/lang_ids.json'
labels_path = 'data/small_inverted_index.json'
type_path = 'data/small_type_counts.json'

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data




def flatten_dict(d):
    if type(d)==str or type(d)==int:
        return [str(d)]
    res = [[i]+flatten_dict(d[i]) for i in d]
    return [j for i in res for j in i]

def count_labels():
    labels = read_json(type_path)
    num_labels = len([i for i in flatten_dict(labels) if not i.isnumeric()])
    return labels,num_labels

def get_label_maps():
    labels,num_labels = count_labels()
    id2label = {}
    A = np.zeros((num_labels,num_labels))
    M = np.eye(num_labels)
    def traverse(d,ancestors = None,count=0):
        if type(d)==int: return count
        for i in d:
            id2label[count] = i.replace('.txt','')
            if ancestors:
                A[ancestors[-1]][count]=1
                for j in ancestors:
                    try:
                        M[j][count] = 1
                    except:
                        import pdb;pdb.set_trace()
                count = traverse(d[i],ancestors+[count],count+1)
            else:
                count = traverse(d[i],[count],count+1)
        return count
    highest = traverse(labels)
    assert num_labels==len(id2label)
    assert max(id2label.keys())+1==highest
    
    return A,M,labels,id2label,num_labels 


def get_data(lang='en'):
    #Load lang_title_articles
    lang_articles_path=articles_path.replace('en',lang)
    articles = read_json(lang_articles_path)

    #Load lang_id lookup
    
    ids = read_json(id_path)
    ids = ids[lang+'wiki']

    #Load id label lookup
    
    labels = read_json(labels_path)

    data = []
    for title,content in articles.items():
        id = ids[title]
        label = labels[id[0]]
        data.append([id,lang,title,content,label])
    return data






# def flatten_label(label): return set(list([j.replace('.txt','') for i in label for j in i]))


#Pytorch Dataset
class EntityDataset(Dataset):
    def __init__(self, data, A, id2label, tokenizer, max_len=512, flatten_label=True):
        self.data = data
        self.A=A
        self.id2label = id2label

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.flatten_label = flatten_label
        self.num_labels = len(id2label)
        self.root_classes = np.where(np.sum(self.A,axis=0)==0)[0]
        print('Root classes: ',[id2label[i] for i in self.root_classes])
        
    
    def match_labels(self,path):
        get_ids = lambda x:[i for i,v in self.id2label.items() if v==x]
        verify_path = lambda path:sum([self.A[i][j] for i,j in zip(path[:-1],path[1:])])==len(path)-1
        
        candidate_ids = [get_ids(i.replace('.txt','')) for i in path]
        candidate_ids[0] = [i for i in candidate_ids[0] if i in self.root_classes]
        assert len(candidate_ids[0])==1

        candidate_paths = list(itertools.product(*candidate_ids))
        results  = [verify_path(path) for path in candidate_paths]
        try: assert sum(results)==1
        except:
            print(path)
            print(candidate_ids)
            for i,j in zip(candidate_paths,results):
                if j:
                    print(i)
                    print([self.id2label[k] for k in i])
        return candidate_paths[results.index(1)]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        id,lang,title,content,label =self.data[item]
        label = [self.match_labels(tuple(i)) for i in label]

        if self.flatten_label:
            label = [j for i in label for j in i]

        labels = torch.zeros(self.num_labels)
        labels[label] = 1
        label = labels
        encoding = self.tokenizer.encode_plus(
            title+content,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors='pt',
            truncation=True
        )

        return {"input_ids":encoding['input_ids'],"labels":label}


def create_data_loader(ds, batch_size=8):
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True
    )

    
if __name__=="__main__":
    A,M,labels,id2label,num_labels = get_label_maps()
    data = get_data(lang='en')

    from load_model import load_tokenizer
    tokenizer=load_tokenizer('bert-base-multilingual-cased')
    ds = EntityDataset(data,A,id2label,tokenizer)
    training_loader = create_data_loader(ds)
    for i in tqdm(training_loader):
        pass
    import pdb;pdb.set_trace()



