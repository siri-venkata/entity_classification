import torch
import torch.nn as nn
import torch.nn.functional as F
if __name__=="__main__":
    from ..utils.utils import seed_all
    seed_all(42)
    from .TextEncoder import load_text_encoder
else:
    from models.TextEncoder import load_text_encoder

class Transformer(nn.Module):
    def __init__(self,tokenizer,d_model,nhead,num_layers=3,numlabels=2,ndim1=2048,ndim2=1024):
        super(Transformer, self).__init__()
        # self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        # self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True), num_layers=num_layers)
#        self.mlp =nn.Sequential(
#                                nn.Linear(d_model,ndim1),
#                                nn.LayerNorm(ndim1),
#                                nn.ReLU(),
#                                nn.Linear(ndim1,ndim2),
#                                nn.LayerNorm(ndim2),
#                                nn.ReLU(),
#                                nn.Linear(ndim2,numlabels),
#                                )
        self.mlp = nn.Linear(d_model,numlabels)
        self.tokenizer = tokenizer

    def forward(self, **inputs):
        # max_size = max([i.shape[0] for i in x])
        # x = torch.stack([F.pad(i,(0,0,0,max_size-i.shape[0]),"constant",0) for i in x])
        x = self.transformer_encoder(**inputs)
        x = F.relu(x)        
        x = self.mlp(x[:,0,:])
        return x.reshape(x.shape[0],-1)
    
class HierarchicalTransformer(nn.Module):
    def __init__(self,encoder_model,classifier_model,):
        super(HierarchicalTransformer, self).__init__()
        self.encoder_model = encoder_model
        self.classifier_model = classifier_model
        
    
    def forward(self, **input_ids):
        x = self.encoder_model(**input_ids)
        mask = (input_ids['outer_attention_mask']<1).to(x.device)
        x = self.classifier_model(src = x, src_key_padding_mask = mask)
        
        return x

def load_hierarchial_model(label_graph,args):
    text_encoder = load_text_encoder(args)
    tokenizer = text_encoder.tokenizer
    classifier_model = Transformer(tokenizer,args.d_model,args.nhead,args.num_layers,args.num_labels)
    model = HierarchicalTransformer(text_encoder,classifier_model)
    return model


if __name__=="__main__":
    model = Transformer(768,8)
  
