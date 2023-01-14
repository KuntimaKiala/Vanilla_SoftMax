import torch 
from torch import nn, optim

class Trainer(nn.Module) :
    
    def __init__(self, model, epochs=2, learning_rate=0.01, momentum=0.9) -> None:
        super().__init__()
        self.epochs        = epochs
        self.learning_rate = learning_rate
        self.momentum      = momentum
        self.optimizer     = optim.SGD(model.parameters(), lr=self.learning_rate, momentum=self.momentum)
        self.loss_fn       = nn.CrossEntropyLoss()
        self.model         = model
    
    def train(self, data) :
        self.model.train()
        for X, y in data :
            self.optimizer.zero_grad
            y_hat = self.model(X) #
            self.loss = self.loss_fn(y_hat, y)
            self.loss.backward()
            self.optimizer.step()
    
    def test(self, data) :
        precision, loss, num_batches, size = 0, 0, len(data), len(data.dataset)
        self.model.eval()
        with torch.no_grad() :
            for X, y in data :
                y_hat = self.model(X) #
                loss += self.loss_fn(y_hat, y).item()
                precision += (y_hat.argmax(dim=1) == y).type(torch.float).sum(axis=0).item()
        loss      /= num_batches
        precision /= size
        print(f"Test Error: \n Accuracy: {(100*precision):>0.1f}%, Avg loss: {loss:>8f} \n")
        return loss, precision
    
    def save(self, epoch, loss, precision, path) :
        
        torch.save({'epoch': epoch,
                    'precision': precision,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss,}, path)
        
    def run(self,train_data, test_data) :
        
        for epoch in range(self.epochs) :
            self.train(train_data)
            loss, precision = self.test(test_data)
            path = "/checkpoints/checkpoint_" + str(epoch) + ".cpkt"
            self.save(epoch=epoch, loss=loss, precision=precision, path=path)