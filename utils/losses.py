import torch
import torch.nn.functional as F

def add_variable_to_scope(**kwargs):
    def decorator(func):
        def wrapper(*args, **func_kwargs):
            result = func(*args, **func_kwargs,**kwargs)
            return result
        return wrapper
    return decorator






bceloss = torch.nn.BCELoss()
bcell = torch.nn.BCEWithLogitsLoss()
def BCELoss(outputs, targets,**kwargs):
    #outputs = torch.sigmoid(outputs)
    return outputs,bcell(outputs,targets)

def CBCELoss(outputs, targets,**kwargs):

    outputs = torch.sigmoid(outputs)
    outputs = outputs.reshape(outputs.shape[0],1,-1)
    M = kwargs["M"]
    # Repeat  outputs to make  square matrix
    H = outputs.repeat(1,outputs.shape[2],1)

    # Hadamard product M and H
    MCM_ = torch.mul(H,M)
    MCM = torch.max(MCM_,dim=2).values
    if targets is None:
        return MCM, None
    loss = bceloss(torch.sigmoid(outputs),targets)
    return MCM, loss




#Custom Loss function
def MCLossold(outputs, targets, **kwargs):
    #Apply sigmoid to outputs
    outputs = torch.sigmoid(outputs)
    outputs = outputs.reshape(outputs.shape[0],1,-1)
    M = kwargs["M"]
    # Repeat  outputs to make  square matrix
    H = outputs.repeat(1,outputs.shape[2],1)

    # Hadamard product M and H
    MCM_ = torch.mul(H,M)
    MCM = torch.max(MCM_,dim=2).values

    if targets is None:
        return MCM, None
    # Make conditional outputs matrix
    mul = torch.mul(outputs,targets.reshape(targets.shape[0],1,-1))
    H_ = mul.repeat(1,mul.shape[2],1)

    # Compute constrained outputs based on targets
    part_a = torch.mul(1-targets,MCM)
    part_b = torch.mul(targets,torch.max(torch.mul(M,H_),dim=2).values)
    constrained_outputs = part_a + part_b

    # Compute loss
    loss = bceloss(constrained_outputs,targets)


    return MCM, loss


def MCLoss(outputs, targets, **kwargs):
    if targets is None:
        return outputs, None
    multiplier = 1
#    R =multiplier* kwargs["R"]
#
#    #Apply sigmoid to outputs
#    #outputs = torch.sigmoid(outputs)
#    outputs_ = outputs.reshape(outputs.shape[0],-1,1)
#
#
#    
#    targets_ = targets.reshape(targets.shape[0],-1,1)
#    tar = targets_.repeat(1,1,targets_.shape[1])
#    # risk_ = 1+R.matmul(targets_)
#    risk_ = 2*multiplier - torch.max(torch.mul(R,tar),dim=2).values
#    risk_ = risk_.reshape(risk_.shape[0],-1,1)
    part_a = -torch.mul(targets_,F.logsigmoid(outputs_))
    part_b = -torch.mul(torch.mul(risk_,1-targets_),F.logsigmoid(1-outputs_))
    loss = multiplier* part_a + part_b
    return outputs, torch.sum(loss)


def RACLoss(outputs, targets, **kwargs):
    if targets is None:
        return outputs, None
    multiplier = 100
    R =multiplier* kwargs["R"]

    #Apply sigmoid to outputs
    #outputs = torch.sigmoid(outputs)
    outputs_ = outputs.reshape(outputs.shape[0],-1,1)


    
    targets_ = targets.reshape(targets.shape[0],-1,1)
    tar = targets_.repeat(1,1,targets_.shape[1])
    # risk_ = 1+R.matmul(targets_)
    risk_ = 2*multiplier - torch.max(torch.mul(R,tar),dim=2).values
    risk_ = risk_.reshape(risk_.shape[0],-1,1)
    part_a = -torch.mul(targets_,F.logsigmoid(outputs_))
    part_b = -torch.mul(torch.mul(risk_,1-targets_),F.logsigmoid(1-outputs_))
    loss = multiplier* part_a + part_b
    return outputs, torch.sum(loss)

def DACLoss(outputs, targets, **kwargs):
    if targets is None:
        return outputs, None
    multiplier = 100
    D = multiplier*kwargs["D"]
    #Apply sigmoid to outputs
    #outputs = torch.sigmoid(outputs)
    outputs_ = outputs.reshape(outputs.shape[0],-1,1)
    targets_ = targets.reshape(targets.shape[0],-1,1)
    tar = targets_.repeat(1,1,targets_.shape[1])

    
    severity_ = multiplier+torch.min(torch.mul(D,tar),dim=2).values
    severity_ = severity_.reshape(severity_.shape[0],-1,1)
    
    part_a = -torch.mul(targets_,F.logsigmoid(outputs_))
    part_b = -torch.mul(torch.mul(severity_,1-targets_),F.logsigmoid(1-outputs_))
    loss = multiplier* part_a + part_b
    return outputs, torch.sum(loss)
    
def RADACLoss(outputs,targets,**kwargs):
    if targets is None:
        return outputs, None
    multiplier=100
    D = multiplier*kwargs["D"]
    R = multiplier*kwargs["R"]
    #outputs = torch.sigmoid(outputs)
    outputs_ = outputs.reshape(outputs.shape[0],-1,1)
    targets_ = targets.reshape(targets.shape[0],-1,1)
    tar = targets_.repeat(1,1,targets_.shape[1])

    risk_ = torch.max(torch.mul(R,tar),dim=2).values
    risk_ = risk_.reshape(risk_.shape[0],-1,1)

    severity_ = torch.min(torch.mul(D,tar),dim=2).values
    severity_ = severity_.reshape(severity_.shape[0],-1,1)

    multiplier = 3*multiplier+severity_-risk_
    part_a = -torch.mul(targets_,F.logsigmoid(outputs_))
    part_b = -torch.mul(torch.mul(multiplier,1-targets_),F.logsigmoid(1-outputs_))
    loss = 2*multiplier* part_a + part_b
    return outputs, torch.sum(loss)








class Lossaggregator():
    def __init__(self,batch_size=8):
        self.losses = []
        self.batch_size = batch_size
    def add(self,loss):
        self.losses.append(loss)
    
    def reset(self):
        self.losses = []

    def get(self):
        return torch.mean(torch.Tensor(self.losses))/self.batch_size



loss_functions={"bce":BCELoss,"mc":MCLoss,'cbc':CBCELoss,'rac':RACLoss,'dac':DACLoss,'radac':RADACLoss}

if __name__=="__main__":
    B, n = 2,3

    # Make an upper triangular matrix of shape nxn
    M = torch.triu(torch.ones(n,n),diagonal=0)
    R = torch.randn(n,n)
    
    #Make a random tensor of shape Bxn
    outputs = torch.rand(B,n)

    #Make a random zeros and ones tensor of shape Bxn
    targets = torch.randint(0,2,(1,n))
    targets = targets.repeat(B,1)*1.0

    print(outputs.shape,targets.shape)
    MCLoss = add_variable_to_scope(M=M,D=R)(loss_functions["dac"])

    MCM, loss = MCLoss(outputs,targets)

    print(MCM.shape)
    print(loss,loss.item())

    

