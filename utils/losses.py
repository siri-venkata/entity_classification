import torch



bceloss = torch.nn.BCELoss()

def BCELoss(outputs, targets,M):
    return torch.sigmoid(outputs),bceloss(outputs,targets)

#Custom Loss function
def MCLoss(outputs, targets, M):
    #Apply sigmoid to outputs
    outputs = torch.sigmoid(outputs)
    outputs = outputs.reshape(outputs.shape[0],1,-1)

    # Repeat  outputs to make  square matrix
    H = outputs.repeat(1,outputs.shape[2],1)

    # Hadamard product M and H
    MCM_ = torch.mul(H,M)
    MCM = torch.max(MCM_,dim=2).values

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



loss_functions={"bce":BCELoss,"mc":MCLoss}

if __name__=="__main__":
    B, n = 2,3

    # Make an upper triangular matrix of shape nxn
    M = torch.triu(torch.ones(n,n),diagonal=0)
    
    #Make a random tensor of shape Bxn
    outputs = torch.rand(B,n)

    #Make a random zeros and ones tensor of shape Bxn
    targets = torch.randint(0,2,(1,n))
    targets = targets.repeat(B,1)*1.0

    MCM, loss = MCLoss(outputs,targets,M)

    print(MCM.shape)
    print(loss,loss.item())

    

