import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import entropy
from torchvision import transforms
import torchvision.datasets as datasets


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
        if self.dataset_name == '/home/phd/datasets/CURIAL':
            # Assume data is loaded from files in a specific format for CURIAL datasets
            # Here we just create dummy data for demonstration purposes
            #data = np.random.rand(1000, 10)  # Replace with actual data loading FOR INITIAL CODE CHECKING......
            #targets = np.random.randint(0, 2, 1000)  # Replace with actual targets
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(y_test)
            self.train_dataset = FederatedDataset(X_train, y_train)
            self.test_dataset = FederatedDataset(X_test, y_test)
        elif self.dataset_name == '/home/phd/datasets/MIMIC-III':
            # Similar to CURIAL, assume proper data loading and preprocessing
            #data = np.random.rand(2000, 15)  # Replace with actual data FOR INITIAL CODE CHECKING......
            #targets = np.random.randint(0, 5, 2000)  # Replace with actual targets
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(y_test)
            self.train_dataset = FederatedDataset(X_train, y_train)
            self.test_dataset = FederatedDataset(X_test, y_test)
        elif self.dataset_name == '/home/phd/datasets/UCIHeartDisease':
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
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            train_size = int(0.8 * len(full_dataset))
            val_size = len(full_dataset) - train_size
            self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
            self.test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        elif self.dataset_name == '/home/phd/datasets/BrainTumorSegmentation':
            # Assume proper data loading and preprocessing for BraTS
            #data = np.random.rand(500, 20)  # Replace with actual data FOR INITIAL CODE CHECKING......
            #targets = np.random.randint(0, 3, 500)  # Replace with actual targets
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(y_test)
            self.train_dataset = FederatedDataset(X_train, y_train)
            self.test_dataset = FederatedDataset(X_test, y_test)
        elif self.dataset_name == '/home/phd/datasets/PathMNIST':
            # Assume proper data loading and preprocessing for PathMNIST
            #data = np.random.rand(10000, 15)  # Replace with actual data FOR INITIAL CODE CHECKING......
            #targets = np.random.randint(0, 9, 10000)  # Replace with actual targets
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(y_test)
            self.train_dataset = FederatedDataset(X_train, y_train)
            self.test_dataset = FederatedDataset(X_test, y_test)
        elif self.dataset_name == '/home/phd/datasets/MIT-BIHArrhythmia':
            # Assume proper data loading and preprocessing for MIT-BIH Arrhythmia
            #data = np.random.rand(800, 12)  # Replace with actual data FOR INITIAL CODE CHECKING......
            #targets = np.random.randint(0, 5, 800)  # Replace with actual targets
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(y_test)
            self.train_dataset = FederatedDataset(X_train, y_train)
            self.test_dataset = FederatedDataset(X_test, y_test)
        elif self.dataset_name == '/home/phd/datasets/PediatricMRI':
            # Assume proper data loading and preprocessing for Pediatric MRI
            #data = np.random.rand(600, 18)  # Replace with actual data ..FOR INITIAL CODE CHECKING......
            #targets = np.random.randint(0, 3, 600)  # Replace with actual targets
            X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2, random_state=42)
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
            X_train = torch.FloatTensor(X_train)
            X_test = torch.FloatTensor(X_test)
            y_train = torch.LongTensor(y_train)
            y_test = torch.LongTensor(y_test)
            self.train_dataset = FederatedDataset(X_train, y_train)
            self.test_dataset = FederatedDataset(X_test, y_test)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def prepare_data(self, batch_size=32):
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        if self.val_dataset:
            self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)