import torch
import torch.nn as nn
import torch.optim as optim


class Client:
    def __init__(self, client_id, train_loader, val_loader, test_loader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = self._create_model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def _create_model(self): # replace with mednet (publishing.... after finalizing the HP)
        # Create a simple neural network model based on the dataset
        if isinstance(self.train_loader.dataset, FederatedDataset) and self.train_loader.dataset.data.shape[1] == 10:
            model = nn.Sequential(
                nn.Linear(10, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
        elif isinstance(self.train_loader.dataset, FederatedDataset) and self.train_loader.dataset.data.shape[1] == 15:
            model = nn.Sequential(
                nn.Linear(15, 128),
                nn.ReLU(),
                nn.Linear(128, 5)
            )
        elif isinstance(self.train_loader.dataset, FederatedDataset) and self.train_loader.dataset.data.shape[1] == 20:
            model = nn.Sequential(
                nn.Linear(20, 128),
                nn.ReLU(),
                nn.Linear(128, 3)
            )
        else:
            model = nn.Sequential(
                nn.Linear(784, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        return model

    def train(self, num_epochs=1):
        criterion = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()

    def evaluate(self, loader):
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        loss = 0
        with torch.no_grad():
            for data, target in loader:
                output = self.model(data)
                loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = correct / total
        avg_loss = loss / len(loader)
        return accuracy, avg_loss