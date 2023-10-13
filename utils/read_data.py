import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset,Dataset, DataLoader
import functools
import itertools
from more_itertools import chunked
from tqdm import tqdm
if __name__=="main":
    from .utils import read_json,flatten_dict,flatten_list,AttrDict
    from .load_model import load_base_model,load_tokenizer
else:
    from utils.utils import read_json,flatten_dict,flatten_list,AttrDict
    from utils.load_model import load_base_model,load_tokenizer


articles_path ='data/articles/enwiki.json'
id_path = 'data/articles/lang_ids.json'
labels_path = 'data/small_inverted_index.json'
type_path = 'data/small_type_counts.json'

def get_paths_ali(args):
    articles_path = args.data+'/articles/enwiki.json'
    id_path = args.data+'/articles/lang_ids.json'
    labels_path = args.data+'/small_inverted_index.json'
    return articles_path,labels_path,id_path

def get_paths_t(args):
    type_path = args.data+'/small_type_counts.json'
    return type_path

def get_paths_cs(args):
    prob_path = 'labels/conditional_dict.json'
    dist_path = 'labels/shortest_paths.json'
    return prob_path,dist_path

def count_labels(args):
    type_path = get_paths_t(args)
    labels = read_json(type_path)
    num_labels = len(set([i.replace('.txt','') for i in flatten_dict(labels) if not i.isnumeric()]))
    print('Number of labels: ',num_labels)
    return labels,num_labels

def get_risk_severity(args,label2id):
    prob_path,dist_path = get_paths_cs(args)
    prob = read_json(prob_path)
    dist = read_json(dist_path)

    R = np.zeros((len(prob),len(prob)))
    for i in prob:
        for j in prob[i]:
            R[label2id[i]][label2id[j]]=prob[i][j]
    
    D = np.zeros((len(prob),len(prob)))
    for i in dist:
        if i.isnumeric(): continue
        for j in dist[i]:
            if j.isnumeric(): continue
            D[label2id[i]][label2id[j]]=dist[i][j]
    
    for i in range(len(D)):
        summ = np.mean([D[i][j] for j in range(len(D)) if D[i][j]<2000])
        for j in range(len(D)):
            if D[i][j]>=2000:
                D[i][j]=1
            else:
                D[i][j]=D[i][j]/(summ+0.01)


    return R,D

def get_label_maps(args):
    labels,num_labels = count_labels(args)
    id2label = {}
    label2id = {}

    A = np.zeros((num_labels,num_labels))
    M = np.eye(num_labels)
    def traverse(d,ancestors = None,count=0):
        if type(d)==int: return count
        for i in d:
            unlabelled = 1 if i.replace('.txt','') not in label2id else 0
            node_id = count if unlabelled else label2id[i.replace('.txt','')]
            id2label[node_id] = i.replace('.txt','')
            label2id[i.replace('.txt','')] = node_id

            if ancestors:
                A[ancestors[-1]][node_id]=1
                for j in ancestors:
                    try:
                        M[j][node_id] = 1
                    except:
                        import pdb;pdb.set_trace()
                count = traverse(d[i],ancestors+[node_id],count+1 if unlabelled else count)
            else:
                count = traverse(d[i],[node_id],count+1 if unlabelled else count)
        return count
    highest = traverse(labels)

    assert num_labels==len(id2label)
    assert max(id2label.keys())+1==highest

    R,D = get_risk_severity(args,label2id)
    
    return A,M,R,D,labels,label2id,id2label,num_labels

def verify_if_DAG(A):
    import networkx as nx
    G = nx.DiGraph(A)
    try:
        assert nx.is_directed_acyclic_graph(G)
    except:
        raise Exception('Graph is not a DAG')
    return True

def save_label_graph(A,id2label):
    import networkx as nx
    import matplotlib.pyplot as plt
    G = nx.DiGraph(A)
    nx.draw(G,labels=id2label,with_labels=True)

    #Store graph and node names as gexf
    nx.relabel_nodes(G,id2label,copy=False)
    nx.write_gexf(G,'labels/label_graph.gexf')

class MissingDataError(Exception):
    def __init__(self, message=None):
        self.message = message

    def __str__(self):
        return 'Some data is missing' if self.message is None else 'Follwoing data is missing: '+self.message
    
def get_predict_data(args):
    all_lang_articles = read_json(args.predict_file)
    data = []
    inex=0
    for lang in all_lang_articles:
        articles = all_lang_articles[lang]
        for title,content in articles.items():
            data.append([index,lang,title,content,0])
            index+=1
    return data
            

def get_data(args,articles_path='',lang='en',load_label=True):
    #Load lang_title_articles
    if not articles_path:
        articles_path,labels_path,id_path = get_paths_ali(args)

    langs = "aswiki|bnwiki|guwiki|hiwiki|knwiki|mlwiki|mrwiki|orwiki|pawiki|tawiki|tewiki|enwiki".replace('wiki','').split('|')
    lang_index = langs.index(lang)

    lang_articles_path=articles_path.replace('en',lang)
    articles = read_json(lang_articles_path)

    #Load lang_id lookup
    if load_label:
        labels = read_json(labels_path)
        ids = read_json(id_path)
        ids = ids[lang+'wiki']

    #Load id label lookup
    data = []
    missing_ids = []
    missing_titles = []
    missing_labels = []

    for title,content in articles.items():
        if load_label:
            try:
                id = ids[title]
            except:
                missing_titles.append(title)
                continue
                
            label = labels[id[0]]
            data.append([int(id[0][1:]),lang_index,title,content,label])
        else:
            data.append([int(id[0][1:]),lang_index,title,content])

    # print('Missing titles: ',len(missing_titles),' out of ',len(articles))
    # print('Present Titles ', set(articles.keys())-set(missing_titles))

    for title,id in ids.items():
        try:
            articles[title]
        except:
            missing_ids.append(id)

    # print('Missing ids from id list ',len(missing_ids),' out of ',len(ids))

    try:
        if len(missing_titles)+len(missing_ids)>0:
            raise MissingDataError("Titles: "+str(len(missing_titles))+" Ids: "+str(len(missing_ids)))
    except MissingDataError as e:
        print(e)


    return data




def load_label_graph(A,id2label,args):
    if args.init_label_nodes=='mbert':
        model = load_base_model(AttrDict({"model_name":args.model_name,'model_path':''}))
        tokenizer = load_tokenizer(AttrDict({"model_name":args.model_name,'model_path':''}),use_fast=False)
        labels = [[i,v] for i,v in id2label.items()]
        label_texts = [v for i,v in id2label.items()]
        label_texts = [i.replace('.txt','') for i in label_texts]
        input_ids = tokenizer(label_texts, return_tensors="pt",padding=True).input_ids
        model.to(args.device)
        input_ids = input_ids.to(args.device)
        output = model(input_ids)
        label_embeddings = output.pooler_output.detach().cpu()
    
    elif args.init_label_nodes=='random':
        label_embeddings = torch.rand((len(id2label),args.d_model)).detach().cpu()
    else:
        raise Exception('Invalid init_label_nodes')
    
    edge_index = torch.LongTensor([[index_i,index_j] for index_i,i in enumerate(A) for index_j,j in enumerate(i) if j==1 ]).reshape(2,-1)

    return AttrDict({'nodes':label_embeddings,'edges':edge_index})





# def flatten_label(label): return set(list([j.replace('.txt','') for i in label for j in i]))


#Pytorch Dataset
class EntityDataset(Dataset):
    def __init__(self, data, A, label2id, id2label, tokenizer, max_len=512, flatten_label=True,has_label=True):
        self.data = data
        self.A=A
        self.label2id = label2id
        self.id2label = id2label

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.flatten_label = flatten_label
        self.num_labels = len(id2label)
        self.root_classes = np.where(np.sum(self.A,axis=0)==0)[0]
        self.has_label = has_label
        
    
    def repeated_names_match_labels(self,path):
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
    
    def match_labels(self,leaf_node):
        '''
        Used to find a path till root class from the leaf node
        '''
        parents = [index for index,i in enumerate(self.A[:,leaf_node]) if i==1]
        return [leaf_node,[self.match_labels(parent) for parent in parents]]


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        if self.has_label:
            id,lang,title,content,label =self.data[item]
            label = [self.match_labels(self.label2id[i[-1].replace('.txt','')]) for i in label]

            if self.flatten_label:
                label = flatten_list(label)
                

            labels = torch.zeros(self.num_labels)
            labels = labels.scatter_(0,torch.LongTensor(label),1)

            label = labels
        else:
            title,content =self.data[item]
        
        encoding = self.tokenizer.encode_plus(
            title+content,
            add_special_tokens=True,
            # max_length=self.max_len,
            # padding="max_length",
            return_tensors='pt',
            truncation=True
        )
        
        return {"input_ids":encoding['input_ids'],"example_id":id,"lang":lang,"labels":label} if self.has_label else {"input_ids":encoding['input_ids']}

def naive_collator(batch,args):
    input_ids,attention_mask = [],[]
    for i in batch:
        content = i['input_ids'].tolist()[0]
        if len(content)>args.max_seq_length:
            data = torch.LongTensor(content[:args.max_seq_length])
            atm = torch.LongTensor([1]*args.max_seq_length)
        else:
            plen = args.max_seq_length-len(content)
            padding = [args.pad_token_id]*plen
            data = torch.LongTensor(content+padding)
            atm = torch.LongTensor(([1]*len(content)+[0]*plen))
        input_ids.append(data)
        attention_mask.append(atm)

    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    # input_ids = torch.LongTensor([+[pad_token_id]*(max_size-i['input_ids'].shape[0]))[:max_size] for i in batch])
    # attention_mask = torch.Tensor([([1]*i['input_ids'].shape[0]+[0]*(max_size-i['input_ids'].shape[0]))[:max_size] for i in batch])
    labels = torch.stack([i['labels'] for i in batch])
    # example_ids = torch.stack([i['example_id'] for i in batch])
    # lang = torch.stack([i['lang'] for i in batch])
    meta_data = torch.LongTensor([[i['example_id'],i['lang']] for i in batch])
    return {"input_ids":input_ids,"labels":labels,
            'meta_data':meta_data,
            'attention_mask':attention_mask}


def hier_collator(batch,args):
    model_inputs = []
    inner_attention_mask =[]
    nchunks =[]
    labels = torch.stack([i['labels'] for i in batch])
    for sample in batch:
        i = sample['input_ids'][0]
        nchunk=0
        for chunk in chunked(i, args.max_seq_length):
            plen = args.max_seq_length-len(chunk)
            model_inputs.append(chunk+[args.pad_token_id]*(plen))
            inner_attention_mask.append(([1]*len(chunk)+[0]*plen))
            nchunk+=1
        nchunks.append(nchunk)
    # print('Number of chunks: ', nchunks)
    model_inputs = torch.LongTensor(model_inputs)
    model_inputs = model_inputs.split(nchunks,dim=0)

    inner_attention_mask = torch.Tensor(inner_attention_mask)
    inner_attention_mask = inner_attention_mask.split(nchunks,dim=0)

    max_size = max(nchunks)
    max_size = min(max_size,args.max_chunks)
    model_inputs = [i[:max_size] for i in model_inputs]
    outer_attention_mask = torch.Tensor([([1]*i.shape[0]+[0]*(max_size-i.shape[0]))[:max_size] for i in model_inputs])

    model_inputs = torch.stack([F.pad(i,(0,0,0,max_size-i.shape[0]),"constant",args.pad_token_id) for i in model_inputs])
    inner_attention_mask = torch.stack([F.pad(i,(0,0,0,max_size-i.shape[0]),"constant",0) for i in inner_attention_mask])
    
    # example_ids = torch.stack([i['example_id'] for i in batch])
    # lang = torch.stack([i['lang'] for i in batch])
    meta_data = torch.LongTensor([[i['example_id'],i['lang']] for i in batch])
    return {"input_ids":model_inputs,"labels":labels,'attention_mask':inner_attention_mask,
            'outer_attention_mask':outer_attention_mask,
            'nchunks':torch.LongTensor(nchunks),
            'meta_data':meta_data}


def graph_collator(batch,args):
    model_inputs = []
    inner_attention_mask =[]
    nchunks =[]
    labels = torch.stack([i['labels'] for i in batch])
    for sample in batch:
        i = sample['input_ids'][0]
        nchunk=0
        for chunk in chunked(i, args.max_seq_length):
            plen = args.max_seq_length-len(chunk)
            model_inputs.append(chunk+[args.pad_token_id]*(plen))
            inner_attention_mask.append(([1]*len(chunk)+[0]*plen))
            nchunk+=1
            if nchunk>10:
                break
        nchunks.append(nchunk)
    # print('Number of chunks: ', nchunks)
    model_inputs = torch.LongTensor(model_inputs)
    model_inputs = model_inputs.split(nchunks,dim=0)

    inner_attention_mask = torch.Tensor(inner_attention_mask)
    inner_attention_mask = inner_attention_mask.split(nchunks,dim=0)

    # max_size = max(nchunks)
    # max_size = min(max_size,args.max_chunks)
    # model_inputs = [i[:max_size] for i in model_inputs]
    # outer_attention_mask = torch.Tensor([([1]*i.shape[0]+[0]*(max_size-i.shape[0]))[:max_size] for i in model_inputs])

    model_inputs = torch.stack(model_inputs)
    inner_attention_mask = torch.stack(inner_attention_mask)
    
    # example_ids = torch.stack([i['example_id'] for i in batch])
    # lang = torch.stack([i['lang'] for i in batch])
    meta_data = torch.LongTensor([[i['example_id'],i['lang']] for i in batch])
    return {"input_ids":model_inputs,"labels":labels,'attention_mask':inner_attention_mask,
            'nchunks':torch.LongTensor(nchunks),
            'meta_data':meta_data}


wrapper_dict = {'naive':naive_collator,'hierarchial':hier_collator,'graph':graph_collator}

def create_data_loader(ds,args, batch_size=8,drop_last=True):
    def collate_wrapper(batch):
        return wrapper_dict[args.model_type](batch,args)
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=8,
        shuffle=True,
        drop_last = drop_last,
        collate_fn=collate_wrapper
    )

def get_data_loaders(A,label2id,id2label,tokenizer,args):
    print('Loading Data')
    tds=[]
    eds=[]
    for lang in args.selected_languages:
        data = get_data(args,lang=lang,load_label=True)
        split = int(len(data)*args.train_test_split)
        tds.append(EntityDataset(data[:split],A, label2id, id2label,tokenizer,  max_len=args.max_seq_length, flatten_label=True))
        eds.append( EntityDataset(data[split:], A, label2id, id2label,tokenizer,  max_len=args.max_seq_length, flatten_label=True))

    train_dataset = torch.utils.data.ConcatDataset(tds)
    eval_dataset = torch.utils.data.ConcatDataset(eds)
    if args.limit_train_batches!=-1 : train_dataset = Subset(train_dataset,[i for i in range(args.limit_train_batches )])
    if args.limit_eval_batches!=-1: eval_dataset = Subset(eval_dataset,[i for i in range(args.limit_eval_batches )])
    training_loader = create_data_loader(train_dataset,args,batch_size=args.batch_size)
    eval_loader = create_data_loader(eval_dataset,args, batch_size=args.eval_batch_size)
    print('Length of train split ',len(train_dataset))
    print('Length of test  split ',len(eval_dataset))
    return training_loader,len(train_dataset)//args.batch_size,eval_loader,len(eval_dataset)//args.batch_size


def predict_data_loader(A,label2id,id2label,tokenizer,args):
    predict_data = get_predict_data(args)
    predict_dataset = EntityDataset(predict_data, A, label2id, id2label,tokenizer,  max_len=args.max_seq_length,has_label=False)
    predict_loader = create_data_loader(predict_dataset,args, batch_size=args.batch_size)
    return predict_loader,len(predict_dataset)


    
if __name__=="__main__":
    from .arguments import get_args
    from collections import Counter
    args = get_args()
    A,M,R,D,labels,label2id,id2label,num_labels = get_label_maps(args)

    verify_if_DAG(A)
    # print('Labels form a DAG')
    # save_label_graph(A,id2label)
    # args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # label_graph = load_label_graph(A,id2label,args)
    # for i in label_graph:
    #     print(i,': ',label_graph.__getattr__(i).shape)

    
    tokenizer=load_tokenizer(AttrDict({"model_name":'bert-base-multilingual-cased',"model_path":""}))
    args.pad_token_id = tokenizer.pad_token_id
    languages = "aswiki|bnwiki|guwiki|hiwiki|knwiki|mlwiki|mrwiki|orwiki|pawiki|tawiki|tewiki|enwiki".replace('wiki','').split('|')
    for lang in languages:
        try:
            data = get_data(args,lang =lang)
            # sanity_check_data(data)
            # continue
            print('Len of '+lang + ' data: ',len(data))
            ds = EntityDataset(data,A,label2id,id2label,tokenizer)
            training_loader = create_data_loader(ds,args)
            for i in tqdm(training_loader):
                # for key in i:
                #     print(key,': ',i[key].shape)
                # break
                pass
            
        except MissingDataError as e:
            print(e)
            print('Skipping ',lang)
            continue
    



