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

    def forward(self, x, edge_index, nchunks):
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x, edge_index)
        
        try:
            x = [self.linear(i[:self.num_labels]) for i in torch.split(x,[self.num_labels+j for j in nchunks])]
        except:
            print('Shape of x is ',x.shape)
            print('Shape of edge_index is ',edge_index.shape)
            print('nchunks ',nchunks)
            raise ValueError
        x = torch.cat(x,dim=1).T

        return x
    
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
        #self.label_nodes.requires_grad = False
        self.label_edges = label_graph.edges
        self.num_nodes = self.label_nodes.shape[0]
    
    def gen_nodes_and_edges(self,text_nodes,node_count=0):
        x = torch.cat([self.label_nodes,text_nodes],dim=0)
        extra_edges = [[i,self.num_nodes+j] for i in range(self.num_nodes) for j in range(text_nodes.shape[0])]
        extra_rev_edges = [[j,i] for i,j in extra_edges]
        extra_self_edges = [[self.num_nodes+i,self.num_nodes+i] for i in range(text_nodes.shape[0])]
        extra_edges = torch.LongTensor(extra_edges+extra_rev_edges+extra_self_edges, device=self.label_edges.device).T
        edge_index = torch.cat([self.label_edges,extra_edges],dim=1)+node_count
        return x,edge_index


    def forward(self, **input_ids):
        '''
        input_ids: B,chunks,seq_len
        attention_mask: B,chunks,seq_len
        nchunks: B
        meta_data: B,meta_dim[id,lang]
        '''
        # input_ids = input_ids['input_ids'].squeeze(0)
        # attention_mask = input_ids['attention_mask'].squeeze(0)
        B,C,L = input_ids['input_ids'].shape
        text_embeddings = self.encoder_model(**input_ids)
        #text_embeddings = text_embeddings.squeeze(0)
        ne = self.node_type_embeddings[1].repeat(B,C,1)
        text_nodes = torch.cat([text_embeddings,ne],dim=2)
        xs,edge_indexes = [],[]
        for i in range(B):
            x,edge_index = self.gen_nodes_and_edges(text_nodes[i],node_count=i*(self.num_nodes+text_nodes.shape[1]))
            xs.append(x)
            edge_indexes.append(edge_index)

        
        x = torch.cat(xs,dim=0).to(self.label_nodes.device)
        edge_index = torch.cat(edge_indexes,dim=1).to(self.label_nodes.device)
        x = self.classifier_model(x, edge_index,input_ids['nchunks'])
        return x


def load_graph_model(label_graph,args):
    text_encoder = load_text_encoder(args)
    classifier_model = GraphNetwork(args.graph_dim, args.nhead, args.num_layers, args.num_labels)
    model = LMGNN(text_encoder, classifier_model,label_graph, args.graph_dim,)
    return model

