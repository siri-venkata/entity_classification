from sklearn.metrics import f1_score, roc_auc_score, accuracy_score,roc_curve,average_precision_score
import torch
import numpy as np
import csv
import traceback
from tqdm import tqdm
from utils.utils import flatten_list

class HF1Computer():
    def __init__(self):
        self.TP = 0
        self.prediction_count = 0
        self.label_count = 0
    
    def update_counts(self,predictions,labels):
        self.TP+=len(set(predictions).intersection(set(labels)))
        self.prediction_count+=len(set(predictions))
        self.label_count+=len(set(labels))

    def add(self,predictions,labels):
        for prediction,label in zip(predictions,labels): self.update_counts(prediction,label)

    def reset(self):
        self.TP = 0
        self.prediction_count = 0
        self.label_count = 0

    def hierarchial_precision_(self):
        return self.TP/self.prediction_count if self.prediction_count>0 else 0


    def hierarchial_recall_(self):
        return self.TP/self.label_count if self.label_count>0 else 0

    def hierarchial_f1(self):
            
        HP = self.hierarchial_precision_()
        HR = self.hierarchial_recall_()
        HF1 = 2*HP*HR/(HP+HR) if HP+HR>0 else 0

        if self.label_count==0:
            raise ValueError('Calculated metrics after providing results atleast once')
        return {'HP':HP,'HR':HR,'HF1':HF1}  
    


def tune_threshold(probs,truth):
    thresholds = [10**i for i in range(-5,1)]
    f1_scores = []
    for threshold in thresholds:
        y_pred = torch.zeros(probs.shape)
        y_pred[torch.where(probs >= threshold)] = 1
        f1_scores.append(f1_score(truth, y_pred,average='micro'))
    fpr, tpr, thresholds = roc_curve(truth.reshape(-1,1), probs.reshape(-1,1))
#    print({"fpr":fpr,"tpr":tpr,"thresholds":thresholds})
#    print(thresholds)
    print('Best Threshold: ',thresholds[np.argmax(tpr-fpr)])
    print('**********************')    
    best_threshold = thresholds[np.argmax(tpr-fpr)]
    return best_threshold



def multi_label_metrics(y_pred, y_true,):
    # # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    # probs = torch.sigmoid(predictions) if apply_sigmoid else predictions

    # # next, use threshold to turn them into integer predictions
    # y_pred = np.zeros(probs.shape)
    # y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    metrics = {'f1': f1_micro_average,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, 
            tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

def one_hot_decode(labels,A):
    return [torch.where(row >= 1)[0].tolist() for row in torch.matmul(labels,A.T)]




class Metricsaggregator():
    def __init__(self,A,id2label,name,args,apply_sigmoid=True,threshold=0.5):
        self.predictions = []
        self.labels=[]
        self.meta_data=[]
        self.args = args
        self.A=A
        self.id2label = id2label
        self.apply_sigmoid=apply_sigmoid
        self.threshold = threshold
        self.HF1 = HF1Computer()
        self.pointer = 0
        self.name = name


    def add(self,predictions,labels,meta_data):
        self.predictions.append(predictions.detach().cpu())
        self.labels.append(labels.detach().cpu())
        self.meta_data.append(meta_data)

    def reset(self):
        self.predictions = []
        self.labels = []
        self.meta_data = []
        self.pointer = 0
        self.HF1.reset()

    def get_comparables(self,return_all = False, meta_filter=None):

        if return_all:
            probs = torch.sigmoid(torch.cat(self.predictions)) if self.apply_sigmoid else torch.cat(self.predictions)
            y_true = torch.cat(self.labels)
            meta_data=torch.cat(self.meta_data)
        else:
            try:
                pointer = self.pointer
                probs = torch.sigmoid(torch.cat(self.predictions[pointer:])) if self.apply_sigmoid else torch.cat(self.predictions[pointer:])
                y_true = torch.cat(self.labels[pointer:])
                meta_data=torch.cat(self.meta_data[pointer:])
                self.pointer = len(self.predictions)
            except:
                return None,None,None,None
        #self.threshold = tune_threshold(probs,y_true)
        y_pred = torch.zeros(probs.shape)
        y_pred[torch.where(probs >= self.threshold)] = 1

        

        return y_pred,probs,y_true,meta_data
            
    def get(self,recompute_auc=False):
        try:
            
            if recompute_auc:
                y_pred,probs,y_true,meta_data = self.get_comparables(True)
                roc_auc = roc_auc_score(y_true, probs, average = 'micro')
                prc_auc = average_precision_score(y_true, probs, average = 'micro')
                metrics={'roc_auc':roc_auc,'prc_auc':prc_auc}
                y_pred,probs,y_true,meta_data = self.get_comparables()
                if y_pred is not None:
                    op,ot = one_hot_decode(y_pred,self.A),one_hot_decode(y_true,self.A)
                    self.HF1.add(op,ot)
                metrics2 = self.HF1.hierarchial_f1()
                metrics.update(metrics2)
                return metrics
            else:      
                y_pred,probs,y_true,meta_data = self.get_comparables()
                if y_pred is not None:
                    op,ot = one_hot_decode(y_pred,self.A),one_hot_decode(y_true,self.A)
                    self.HF1.add(op,ot)
                metrics2 = self.HF1.hierarchial_f1()
                return metrics2
        except Exception as e:
            import pdb;pdb.set_trace();
            raise ValueError('Error in metric calculation ')
    
    def save_predictions(self,eval=False,name=''):
        def verbalize_labels(labels):
            return [[self.id2label[j] for j in i] for i in labels]
        import json
        y_pred,probs,y_true,meta_data = self.get_comparables(True)
        
        readable_predictions = [one_hot_decode(y_pred,self.A),one_hot_decode(y_true,self.A)]
        readable_predictions = list(map(verbalize_labels,readable_predictions))
        readable_predictions.extend([1,2])
        readable_predictions[2] = [i[0] for i in meta_data]
        readable_predictions[3] = [i[1] for i in meta_data]

        # final_data = [[int(id),int(lang),"_|_".join(predictions),"_|_".join(labels)] for predictions,labels,id,lang in zip(*readable_predictions)]
        final_data = [[int(id),int(lang),predictions,labels] for predictions,labels,id,lang in zip(*readable_predictions)]
        final_data =[["id","lang","predictions","ground truth"]] + final_data


        split = 'train' if not eval else 'eval'
        with open(self.args.save_model_path+'/predictions'+split+name+'.json','w') as f:
            json.dump(y_pred.detach().tolist(),f)
        with open(self.args.save_model_path+'/predictions'+split+str(name)+'.csv','w',newline='') as f:
            writer = csv.writer(f)
            writer.writerows(final_data)

        
