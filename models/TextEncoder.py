import torch
import torch.nn as nn


from more_itertools import chunked

if __name__=="__main__":
    from ..utils.utils import read_json,AttrDict
    from ..utils.load_model import load_tokenizer,load_base_model
    from .positional_encoding import PositionalEncoding1D
else:
    from utils.utils import read_json,AttrDict
    from utils.load_model import load_tokenizer,load_base_model
    from models.positional_encoding import PositionalEncoding1D
    



class TextEncoder(nn.Module):
    def __init__(self, model,tokenizer,max_seq_length=512,padding_token=0,freeze_backbone=True):
        super(TextEncoder, self).__init__()
        self.model = model
        self.max_seq_length = max_seq_length
        self.padding_token = padding_token
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        self.positional_encoding = PositionalEncoding1D(self.model.config.hidden_size)
        self.tokenizer = tokenizer

    def forward(self, **inputs):
        B,N,L = inputs['input_ids'].shape
        input_ids = inputs['input_ids'].reshape(B*N,-1)
        attention_mask = inputs['attention_mask'].reshape(B*N,-1)
        outputs = self.model(input_ids=input_ids,attention_mask=attention_mask)
        po =  outputs.pooler_output.reshape(B,N,-1)
        pemb = self.positional_encoding(po[0]).repeat(B,1,1)
        return po+pemb


    # def forward(self, input_ids):
    #     # print('Input shape: ', [len(i) for i in input_ids])
    #     model_inputs = []
    #     nchunks =[]
    #     for i in input_ids:
    #         nchunk=0
    #         for chunk in chunked(i, self.max_seq_length):
    #             model_inputs.append(chunk+[self.tokenizer.pad_token]*(self.max_seq_length-len(chunk)))
    #             nchunk+=1
    #         nchunks.append(nchunk)
    #     # print('Number of chunks: ', nchunks)

    #     model_inputs = torch.LongTensor(model_inputs)
    #     # print('Model input shape: ', model_inputs.shape)
    #     output = self.model(model_inputs)
    #     # print(output.keys())
    #     # for i in output.keys():
    #     #     print(i,': ', output[i].shape)
    #     po =  output.pooler_output
    #     po = torch.split(po,nchunks,dim=0)
    #     return [i+self.positional_encoding(i) for i in po]

def load_text_encoder(args):
    model = load_base_model(args)
    tokenizer = load_tokenizer(args)
    return TextEncoder(model,tokenizer,args.max_seq_length,args.pad_token_id,args.freeze_backbone)

def get_dummy_output():
    args = AttrDict({"model_name":'bert-base-multilingual-cased','model_path':''})

    model = load_base_model(args)
    tokenizer = load_tokenizer(args)
    data = read_json('entity_classification/data/articles/enwiki.json')
    keys = list(data.keys())
    def get_example(index):
        key = keys[index]
        return key+data[key]

    texts = [get_example(i) for i in range(4)]

    input_ids = tokenizer(texts).input_ids
    model = TextEncoder(model,max_seq_length=512)

    output = model(input_ids)
    return output

if __name__ == '__main__':
    args = AttrDict({"model_name":'bert-base-multilingual-cased','model_path':'','num_labels':2})

    model = load_base_model(args)
    tokenizer = load_tokenizer(args)
    data = read_json('entity_classification/data/articles/enwiki.json')
    keys = list(data.keys())
    def get_example(index):
        key = keys[index]
        return key+data[key]

    texts = [get_example(i) for i in range(4)]

    input_ids = tokenizer(texts).input_ids


    model = TextEncoder(model,max_seq_length=512)

    output = model(input_ids)
    for i in output:
        print(i.shape)
