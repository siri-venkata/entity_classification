from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch
import numpy as np
    
# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5,apply_sigmoid=False):


    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    probs = torch.sigmoid(predictions) if apply_sigmoid else predictions

    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, probs, average = 'micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result




class Metricsaggregator():
    def __init__(self,args):
        self.predictions = []
        self.labels=[]
        self.args = args
    def add(self,predictions,labels):
        self.predictions.append(predictions.detach().cpu())
        self.labels.append(labels.detach().cpu())
    
    def reset(self):
        self.predictions = []
        self.labels = []

    def get(self):
        try:
            return multi_label_metrics(torch.cat(self.predictions),torch.cat(self.labels))
        except:
            import pdb;pdb.set_trace()
    
    def save_predictions(self,eval=False):
        import json
        preds = torch.cat(self.predictions).numpy().tolist()
        split = 'train' if not eval else 'eval'
        with open(self.args.save_model_path+'/best/predictions'+split+'.json','w') as f:
            json.dump(preds,f)