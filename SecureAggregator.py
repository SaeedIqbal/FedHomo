''' 
The initial code. 
We are just finalizing the below code, after project completion.
Below are snapshots of our part of methodology.

'''
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
            pass
        elif self.dataset_name == 'Brain Tumor Segmentation':
            # Assume proper data loading and preprocessing for BraTS
            pass
        elif self.dataset_name == 'PathMNIST':
            # Assume proper data loading and preprocessing for PathMNIST
            pass
        elif self.dataset_name == 'MIT-BIH Arrhythmia':
            # Assume proper data loading and preprocessing for MIT-BIH Arrhythmia
            pass
        elif self.dataset_name == 'Pediatric MRI':
            # Assume proper data loading and preprocessing for Pediatric MRI
            pass
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def prepare_data(self, batch_size=32):
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        if self.val_dataset:
            self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)


def generate_prime(bits=512):
    """
    Generate a prime number of the specified bit length.
    """
    def is_prime(n, k=128):
        """
        Miller-Rabin primality test.
        """
        if n == 2 or n == 3:
            return True
        if n <= 1 or n % 2 == 0:
            return False
        r, s = 0, n - 1
        while s % 2 == 0:
            r += 1
            s //= 2
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, s, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    while True:
        num = random.getrandbits(bits)
        if is_prime(num):
            return num


def gcd(a, b):
    """
    Calculate the greatest common divisor of a and b using Euclidean algorithm.
    """
    while b!= 0:
        a, b = b, a % b
    return a


def lcm(a, b):
    """
    Calculate the least common multiple of a and b.
    """
    return abs(a * b) // gcd(a, b)


def mod_inverse(a, m):
    """
    Calculate the modular inverse of a modulo m.
    """
    def egcd(a, b):
        if a == 0:
            return (b, 0, 1)
        else:
            g, y, x = egcd(b % a, a)
            return (g, x - (b // a) * y, y)

    g, x, _ = egcd(a, m)
    if g!= 1:
        raise Exception('modular inverse does not exist')
    else:
        return x % m


def paillier_keygen(bits=512):
    """
    Generate Paillier public and private keys.
    """
    p = generate_prime(bits)
    q = generate_prime(bits)
    n = p * q
    lambda_n = lcm(p - 1, q - 1)
    g = n + 1
    mu = mod_inverse(L(pow(g, lambda_n, n ** 2), n), n)
    public_key = (n, g)
    private_key = lambda_n
    return public_key, private_key


def L(u, n):
    """
    L function used in Paillier decryption.
    """
    return (u - 1) // n


class SecureAggregator:
    def __init__(self, bits=512):
        self.public_key, self.private_key = paillier_keygen(bits)
        self.n_squared = self.public_key[0] ** 2

    def encrypt(self, plaintext):
        """
        Encrypt a plaintext using Paillier cryptosystem.
        """
        n, g = self.public_key
        r = random.randint(1, n - 1)
        ciphertext = (pow(g, plaintext, self.n_squared) * pow(r, n, self.n_squared)) % self.n_squared
        return ciphertext

    def aggregate_encrypted(self, encrypted_values):
        """
        Aggregate encrypted values using Paillier's homomorphic property.
        """
        aggregated = 1
        for value in encrypted_values:
            aggregated = (aggregated * value) % self.n_squared
        return aggregated

    def decrypt(self, ciphertext):
        """
        Decrypt a ciphertext using Paillier cryptosystem.
        """
        lambda_n = self.private_key
        n = self.public_key[0]
        mu = mod_inverse(L(pow(self.public_key[1], lambda_n, n ** 2), n), n)
        m = (L(pow(ciphertext, lambda_n, self.n_squared), n) * mu) % n
        return m


class Client:
    def __init__(self, client_id, train_loader, val_loader, test_loader, aggregator):
        self.client_id = client_id
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.aggregator = aggregator
        self.model = self._create_model()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def _create_model(self):
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

    def encrypt_model_update(self):
        # Flatten the model parameters and encrypt them
        model_update = torch.cat([param.view(-1) for param in self.model.parameters()]).tolist()
        encrypted_update = [self.aggregator.encrypt(int(val)) for val in model_update]
        return encrypted_update


class Server:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.clients = []
        self.aggregator = SecureAggregator()
        self.global_model = None

    def register_client(self, client):
        self.clients.append(client)
        if self.global_model is None:
            self.global_model = client.model.state_dict()

    def federated_train(self, num_rounds=10):
        for round in range(num_rounds):
            encrypted_updates = []
            for client in self.clients:
                client.train()
                encrypted_update = client.encrypt_model_update()
                encrypted_updates.append(encrypted_update)
            # Flatten the list of encrypted updates
            flat_encrypted_updates = [item for sublist in encrypted_updates for item in sublist]
            aggregated_update = self.aggregator.aggregate_encrypted(flat_encrypted_updates)
            decrypted_update = self.aggregator.decrypt(aggregated_update)
            # Convert the decrypted update back to model parameters
            num_params = sum(p.numel() for p in self.clients[0].model.parameters())
            decrypted_update_tensor = torch.tensor(decrypted_update).float()
            param_sizes = [p.numel() for p in self.clients[0].model.parameters()]
            param_tensors = torch.split(decrypted_update_tensor, param_sizes)
            global_update_dict = {}
            start = 0
            for name, param in self.global_model.items():
                end = start + param.numel()
                global_update_dict[name] = param_tensors[start:end].view(param.size())
                start = end
            self.global_model = self._update_global_model(self.global_model, global_update_dict)
            for client in self.clients:
                client.model.load_state_dict(self.global_model)

    def _update_global_model(self, old_state_dict, new_state_dict):
        updated_state_dict = {}
        for key in old_state_dict.keys():
            updated_state_dict[key] = old_state_dict[key] + new_state_dict[key]
        return updated_state_dict


if __name__ == "__main__":
    data_handler = DataHandler('UCI Heart Disease')
    data_handler.prepare_data(batch_size=32)
    server = Server(num_clients=3)
    client1 = Client(1, data_handler.train_loader, data_handler.val_loader, data_handler.test_loader, server.aggregator)
    client2 = Client(2, data_handler.train_loader, data_handler.val_loader, data_handler.test_loader, server.aggregator)
    client3 = Client(3, data_handler.train_loader, data_handler.val_loader, data_handler.test_loader, server.aggregator)
    server.register_client(client1)
    server.register_client(client2)
    server.register_client(client3)
    server.federated_train()