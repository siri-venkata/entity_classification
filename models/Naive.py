import torch
import torch.nn as nn
if __name__=="__main__":
    from ..utils.load_model import load_model,load_base_model
else:
    from utils.load_model import load_model,load_base_model


class Naive(nn.Module):
    def __init__(self,model,args,ndim1=2048,ndim2=1024):
        super(Naive, self).__init__()
        self.model = model
        if args.freeze_backbone:
            for name,param in self.model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
        self.ndim1=ndim1
        self.ndim2=ndim2
        self.mlp =nn.Sequential(
                                nn.Linear(model.config.hidden_size,ndim1),
                                nn.LayerNorm(ndim1),
                                nn.ReLU(),
                                nn.Linear(ndim1,ndim2),
                                nn.LayerNorm(ndim2),
                                nn.ReLU(),
                                nn.Linear(ndim2,args.num_labels),
                                )
    
    def forward(self, **inputs):
        bert_otp = self.model(input_ids =  inputs['input_ids'],attention_mask = inputs['attention_mask']).pooler_output
        return self.mlp(bert_otp)


#def load_naive_model(label_graph,args):
#    model = load_model(args)
#    return Naive(model,args.freeze_backbone)

def load_naive_model(label_graph,args):
    model = load_base_model(args)
    return Naive(model,args)
