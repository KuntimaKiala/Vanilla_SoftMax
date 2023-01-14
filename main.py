import torch
from DataLoading import DataHandler
from torchvision import datasets
from vanilla import SoftMax
from trainer import Trainer

if __name__ == '__main__' :
    device = torch.device('cuda0' if torch.cuda.is_available() else 'cpu')
    data_handler = DataHandler(datasets=datasets.FashionMNIST,download=False, shuffle=True, batch_size=1024) 
    training_dataset, validation_dataset = data_handler.dataset()
    training_data, validation_data = data_handler.DataLoader(training_dataset, validation_dataset)
    input_size = next(iter(training_data))[0].shape[2] * next(iter(training_data))[0].shape[3]
    hidden_size = 1
    output_size = 10
    model = SoftMax(input_size, hidden_size, output_size)
    model.to(device=device)
    trainer = Trainer(model=model, epochs=500, learning_rate=0.01, momentum=0.89)
    trainer.run(training_data, validation_data)