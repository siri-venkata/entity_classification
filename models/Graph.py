import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import  GATv2Conv

if __name__=="__main__":
    from ..utils.utils import seed_all
    seed_all(42)
    from .TextEncoder import load_text_encoder
else:
    from models.TextEncoder import load_text_encoder



class GraphNetwork(nn.Module):
    def __init__(self, d_model, nhead, num_layers=7, num_labels=2):
        super(GraphNetwork, self).__init__()
        self.gnn_layers  =  nn.ModuleList([GATv2Conv(d_model, d_model//nhead, heads=nhead, concat=True, dropout=0.1) for i in range(num_layers)])
        self.linear      =  nn.Linear(d_model, 1)
        self.num_labels = num_labels

    def forward(self, x, edge_index):
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x, edge_index)
        x = self.linear(x)
        return x[:self.num_labels,0].reshape(1,-1)
    
class LMGNN(nn.Module):
    def __init__(self, encoder_model, classifier_model,label_graph, d_model):
        super(LMGNN, self).__init__()
        self.encoder_model = encoder_model
        self.classifier_model = classifier_model
        
        
        # self.node_type_embeddings = nn.Embedding(2, d_model - self.encoder_model.model.config.hidden_size)
        self.node_type_embeddings = nn.Parameter(torch.randn(2, d_model - self.encoder_model.model.config.hidden_size))
        nn.init.xavier_normal_(self.node_type_embeddings)
        # le = self.node_type_embeddings(torch.LongTensor([0])).repeat(self.label_nodes.shape[0],1)
        le = self.node_type_embeddings[0].repeat(label_graph.nodes.shape[0],1)
        self.label_nodes = nn.Parameter(torch.cat([label_graph.nodes,le],dim=1))
        self.label_nodes.requires_grad = False
        self.label_edges = label_graph.edges
        self.num_nodes = self.label_nodes.shape[0]

    
    def forward(self, **input_ids):
        # input_ids = input_ids['input_ids'].squeeze(0)
        # attention_mask = input_ids['attention_mask'].squeeze(0)
        text_embeddings = self.encoder_model(**input_ids)
        text_embeddings = text_embeddings.squeeze(0)
        ne = self.node_type_embeddings[1].repeat(text_embeddings.shape[0],1)
        text_nodes = torch.cat([text_embeddings,ne],dim=1)
        x = torch.cat([self.label_nodes,text_nodes],dim=0)
        extra_edges = [[i,self.num_nodes+j] for i in range(self.num_nodes) for j in range(text_nodes.shape[0])]
        extra_rev_edges = [[j,i] for i,j in extra_edges]
        extra_self_edges = [[self.num_nodes+i,self.num_nodes+i] for i in range(text_nodes.shape[0])]
        # print(len(extra_edges))
        # print(len(extra_rev_edges))
        # print(len(extra_self_edges))
        # print(self.label_edges.shape)
        extra_edges = torch.LongTensor(extra_edges+extra_rev_edges+extra_self_edges, device=self.label_edges.device).T
        edge_index = torch.cat([self.label_edges,extra_edges],dim=1)
        edge_index = edge_index.to(text_embeddings.device)
        # print(x.shape)
        # print(edge_index.shape)
        x = self.classifier_model(x, edge_index)
        return x


def load_graph_model(label_graph,args):
    text_encoder = load_text_encoder(args)
    classifier_model = GraphNetwork(args.graph_dim, args.nhead, args.num_layers, args.num_labels)
    model = LMGNN(text_encoder, classifier_model,label_graph, args.graph_dim,)
    return model

