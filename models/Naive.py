import torch
import torch.nn as nn
if __name__=="__main__":
    from ..utils.load_model import load_model
else:
    from utils.load_model import load_model


class Naive(nn.Module):
    def __init__(self,model,freeze_backbone=True):
        super(Naive, self).__init__()
        self.model = model
        if freeze_backbone:
            for name,param in self.model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False

    
    def forward(self, **inputs):
        return self.model(input_ids =  inputs['input_ids'],attention_mask = inputs['attention_mask']).logits


def load_naive_model(label_graph,args):
    model = load_model(args)
    return Naive(model,args.freeze_backbone)