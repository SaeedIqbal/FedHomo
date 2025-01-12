'''
This file is run on a commodity hardware 
Initially, we have a very limited resources, 
so, we just run and compile the code to remove the errors

'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class FederatedDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class DataHandler:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.scaler = StandardScaler()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self._load_dataset()

    def _load_dataset(self):
        if self.dataset_name == 'CURIAL':
            # Assume data is loaded from files in a specific format for CURIAL datasets
            # Here we just create dummy data for demonstration purposes
            data = np.random.rand(1000, 10)  # Replace with actual data loading
	    # Just for code checking, Initially, we have a very limited computation resources, 
	    # So, we just check our code, is it working or not..
            # you can replace this with proper datasets (files)..
            targets = np.random.randint(0, 2, 1000)  # Replace with actual targets
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(y_test)
            self.train_dataset = FederatedDataset(X_train, y_train)
            self.test_dataset = FederatedDataset(X_test, y_test)
        elif self.dataset_name == 'MIMIC-III':
            # Similar to CURIAL, assume proper data loading and preprocessing
            data = np.random.rand(2000, 15)  # Replace with actual data
            targets = np.random.randint(0, 5, 2000)  # Replace with actual targets
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(y_test)
            self.train_dataset = FederatedDataset(X_train, y_train)
            self.test_dataset = FederatedDataset(X_test, y_test)
        elif self.dataset_name == 'UCI Heart Disease':
            # Load UCI Heart Disease Dataset
            data = np.loadtxt('heart.csv', delimiter=',', skiprows=1)
            X = data[:, :-1]
            y = data[:, -1].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(y_test)
            self.train_dataset = FederatedDataset(X_train, y_train)
            self.test_dataset = FederatedDataset(X_test, y_test)
        elif self.dataset_name == 'ChestMNIST':
            # Assume proper data loading and preprocessing for ChestMNIST
	    # Change this according to your setup.........
            pass
        elif self.dataset_name == 'Brain Tumor Segmentation':
            # Assume proper data loading and preprocessing for BraTS
	    # Change this according to your setup.........
            pass
        elif self.dataset_name == 'PathMNIST':
            # Assume proper data loading and preprocessing for PathMNIST
	    # Change this according to your setup.........
            pass
        elif self.dataset_name == 'MIT-BIH Arrhythmia':
            # Assume proper data loading and preprocessing for MIT-BIH Arrhythmia
	    # Change this according to your setup.........
            pass
        elif self.dataset_name == 'Pediatric MRI':
            # Assume proper data loading and preprocessing for Pediatric MRI
	    # Change this according to your setup.........
            pass
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def prepare_data(self, batch_size=32):
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        if self.val_dataset:
            self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)


class Client:
    def __init__(self, client_id, train_loader, val_loader, test_loader):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = self._create_model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        self.time_taken = 0  # To store time taken for training
        self.energy_consumed = 0  # To store energy consumed during training

    def _create_model(self): 	    # here, we are using our previously published model, this is just for checking the code.
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
        import time
        import torch.nn.functional as F
        start_time = time.time()
        criterion = nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            for data, target in self.train_loader:
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                self.optimizer.step()
        end_time = time.time()
        self.time_taken = end_time - start_time
        # Assume energy consumption is proportional to time taken for simplicity
        self.energy_consumed = self.time_taken

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


class Server:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.clients = []
        self.global_model = None
        self.client_scores = [0] * num_clients  # Initialize client scores
        self.alpha_acc = 0.5  # Weights for contributions
        self.alpha_div = 0.3
        self.alpha_comp = 0.2
        self.beta = 0.9  # Client forgetting factor
        self.gamma = 0.1  # Margin for adaptive thresholding
        self.delta = 0.05  # Adjustment factor
        self.participation_rates = [1.0] * num_clients  # Initial participation rates

    def register_client(self, client):
        self.clients.append(client)
        if self.global_model is None:
            self.global_model = client.model.state_dict()

    def federated_train(self, num_rounds=10):
        for round in range(num_rounds):
            # Calculate evaluation scores
            scores = []
            for i, client in enumerate(self.clients):
                acc_with_client, _ = self._evaluate_global_model_with_client(i)
                acc_without_client, _ = self._evaluate_global_model_without_client(i)
                data_diversity = self.calculate_data_diversity(client)
                computational_efficiency = 1 / (client.time_taken * client.energy_consumed)
                score = (self.alpha_acc * (acc_with_client - acc_without_client) +
                         self.alpha_div * data_diversity +
                         self.alpha_comp * computational_efficiency)
                scores.append(score)
                self.client_scores[i] = score

            avg_score = np.mean(scores)
            threshold = avg_score + self.gamma
            selected_clients = [i for i, score in enumerate(scores) if score >= threshold]

            # Calculate participation rates
            selected_scores = [scores[i] for i in selected_clients]
            participation_rates = [score / sum(selected_scores) for score in selected_scores]
            for i, rate in zip(selected_clients, participation_rates):
                self.participation_rates[i] = rate

            # Train selected clients
            for i in selected_clients:
                client = self.clients[i]
                client.train()
                self._update_global_model(client)

            # Update client scores with forgetting factor
            for i, client in enumerate(self.clients):
                acc_with_client, _ = self._evaluate_global_model_with_client(i)
                acc_without_client, _ = self._evaluate_global_model_without_client(i)
                data_diversity = self.calculate_data_diversity(client)
                computational_efficiency = 1 / (client.time_taken * client.energy_consumed)
                new_score = (self.beta * self.client_scores[i] +
                            (1 - self.beta) * (self.alpha_acc * (acc_with_client - acc_without_client) +
                                             self.alpha_div * data_diversity +
                                             self.alpha_comp * computational_efficiency))
                self.client_scores[i] = new_score

            # Dynamically adjust participation rates
            for i in selected_clients:
                client = self.clients[i]
                data_diversity = self.calculate_data_diversity(client)
                computational_efficiency = 1 / (client.time_taken * client.energy_consumed)
                self.participation_rates[i] *= (1 + self.delta * (data_diversity - computational_efficiency))

    def _evaluate_global_model_with_client(self, client_index):
        original_state = self.global_model
        client = self.clients[client_index]
        client.train()
        self._update_global_model(client)
        accuracy, loss = self.evaluate_global_model()
        self.global_model = original_state
        return accuracy, loss

    def _evaluate_global_model_without_client(self, client_index):
        original_state = self.global_model
        client = self.clients[client_index]
        client_state = client.model.state_dict()
        client.model.load_state_dict(self.global_model)
        self._update_global_model(client)
        accuracy, loss = self.evaluate_global_model()
        client.model.load_state_dict(client_state)
        self.global_model = original_state
        return accuracy, loss

    def _update_global_model(self, client):
        client_state = client.model.state_dict()
        for key in self.global_model.keys():
            self.global_model[key] += client_state[key]

    def evaluate_global_model(self):
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()
        loss = 0
        for client in self.clients:
            for data, target in client.test_loader:
                output = client.model(data)
                loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        accuracy = correct / total
        avg_loss = loss / len(self.clients)
        return accuracy, avg_loss

    def calculate_data_diversity(self, client):
        distances = []
        client_data = []
        for data, _ in client.train_loader:
            client_data.append(data)
        client_data = torch.cat(client_data)
        for other_client in self.clients:
            other_data = []
            for data, _ in other_client.train_loader:
                other_data.append(data)
            other_data = torch.cat(other_data)
            if client_data.dim() == 2:  # Tabular data
                distance = torch.mean(torch.cdist(client_data, other_data))
            else:  # Image data (example for images)
                distance = torch.mean(torch.cdist(client_data.view(client_data.size(0), -1),
                                           other_data.view(other_data.size(0), -1))
            distances.append(distance)
        return torch.mean(torch.stack(distances))


if __name__ == "__main__":
    data_handler = DataHandler('UCI Heart Disease')
    data_handler.prepare_data(batch_size=32)
    server = Server(num_clients=3)
    client1 = Client(1, data_handler.train_loader, data_handler.val_loader, data_handler.test_loader)
    client2 = Client(2, data_handler.train_loader, data_handler.val_loader, data_handler.test_loader)
    client3 = Client(3, data_handler.train_loader, data_handler.val_loader, data_handler.test_loader)
    server.register_client(client1)
    server.register_client(client2)
    server.register_client(client3)
    server.federated_train()