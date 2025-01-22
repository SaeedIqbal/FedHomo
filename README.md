# Federated Learning Framework

This repository contains the code for a federated learning framework with a focus on optimizing client participation through dynamic client selection. The framework is designed to work with various datasets and incorporates techniques for secure aggregation and hierarchical prototyping. However, the code is still under development and not complete. Due to restrictions from the project funding agency, we are unable to release the complete code until the project is finished.

## Overview

The federated learning framework consists of several components, including data handling, client-side training, server-side aggregation, and techniques for optimizing client participation. The following sections provide an overview of the main components and their functionalities.

### Data Handling

The `DataHandler` class is responsible for loading different datasets. It supports multiple datasets such as CURIAL, MIMIC-III, UCI Heart Disease, ChestMNIST, Brain Tumor Segmentation, PathMNIST, MIT-BIH Arrhythmia, and Pediatric MRI. It handles data preprocessing, scaling, and splitting into training, validation, and test sets. It uses PyTorch's `DataLoader` for efficient data loading and batching.

```python
class DataHandler:
    def __init__(self, dataset_name):
        # Initialize data handler
        pass

    def _load_dataset(self):
        # Load dataset based on dataset_name
        pass

    def prepare_data(self, batch_size=32):
        # Prepare data loaders for training, validation, and testing
        pass
```

### Client Class

The `Client` class represents a client in the federated learning setup. Each client has its own dataset, model, and optimizer. It is responsible for training its local model and evaluating the model's performance.

```python
class Client:
    def __init__(self, client_id, train_loader, val_loader, test_loader):
        # Initialize client with its ID and data loaders
        pass

    def _create_model(self):
        # Create a neural network model based on the dataset
        pass

    def train(self, num_epochs=1):
        # Train the client's local model
        pass

    def evaluate(self, loader):
        # Evaluate the client's model on a given data loader
        pass
```

### Server Class

The `Server` class manages multiple clients and aggregates their model updates. It also incorporates dynamic client selection and adaptive participation rates.

```python
class Server:
    def __init__(self, num_clients):
        # Initialize server with number of clients
        pass

    def register_client(self, client):
        # Register a client with the server
        pass

    def federated_train(self, num_rounds=10):
        # Perform federated training for a specified number of rounds
        pass

    def _evaluate_global_model_with_client(self, client_index):
        # Evaluate the global model with a specific client's contribution
        pass

    def _evaluate_global_model_without_client(self, peterm-client_index):
        # Evaluate the global model without a specific client's contribution
        pass

    def _update_global_model(self, client):
        # Update the global model with a client's model
        pass

    def evaluate_global_model(self):
        # Evaluate the global model
        pass

    def calculate_data_diversity(self, client):
        # Calculate data diversity of a client
        pass
```

### Secure Aggregation

The `SecureAggregator` class implements the Paillier cryptosystem for secure aggregation of model updates. It includes key generation, encryption, aggregation, and decryption functions.

```python
class SecureAggregator:
    def __init__(self, bits=512):
        # Initialize with Paillier key generation
        pass

    def encrypt(self, plaintext):
        # Encrypt plaintext using Paillier cryptosystem
        pass

    def aggregate_encrypted(self, encrypted_values):
        # Aggregate encrypted values
        pass

    def decrypt(self, ciphertext):
        # Decrypt ciphertext using Paillier cryptosystem
        pass
```

### Hierarchical Prototyping

The `HierarchicalPrototyping` class is responsible for initializing and updating prototypes, calculating adaptive weights, and enforcing consistency between local and global prototypes.

```python
class HierarchicalPrototyping:
        def __init__(self, clients):
            # Initialize hierarchical prototyping with clients
            pass

        def initialize_prototypes(self):
            # Initialize global and local prototypes
            pass

        def calculate_adaptive_weights(self):
            # Calculate adaptive weights for prototypes
            pass

        def update_prototypes(self):
            # Update prototypes based on adaptive weights
            pass

        def enforce_consistency(self, mu=0.1, nu=0.1):
            # Enforce consistency between prototypes
            pass
```

## Usage

Here is a basic example of how to use the framework:

```python
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
```

## Future Work

The following aspects of the code are still under development and will be updated in the future:
- Completion of the `HierarchicalPrototyping` class, including more sophisticated loss calculations and prototype updates.
- Enhancement of the `SecureAggregator` class for better security and performance.
- Further optimization of the dynamic client selection mechanism, including more accurate energy consumption tracking and refined distance metrics for data diversity.
- Integration of additional datasets and models.


## License

This code is developed under the restrictions of the project funding agency, and as such, the full code is not available for public release at this time. We aim to provide more complete and optimized code upon project completion. Please check back for updates.


## Contributing

Contributions are not currently accepted due to the project's funding restrictions. We will update the repository and contribution guidelines once the project is complete.


## Contact

For any inquiries, please contact [saeed.iqbal@szu.edu.cn](mailto:saeed.iqbal@szu.edu.cn).


## References for Datasets

Here are the references for the datasets used in this project:

### UCI Heart Disease Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Description**: This dataset contains various attributes related to heart disease diagnosis. It includes features such as age, sex, cholesterol levels, and more. You can download the dataset directly from the UCI Machine Learning Repository.

### MIMIC-III Dataset
- **Source**: [MIMIC-III Clinical Database](https://mimic.physionet.org/)
- **Description**: A large, freely accessible critical care database. Access to this dataset requires credentialing through PhysioNet. You need to complete a training course and sign a data use agreement to obtain access.

### CURIAL Dataset
- **Source**: [CURIAL Dataset Repository]()
- **Description**: (Please provide the actual source if available) The CURIAL dataset is used for specific research purposes. It may require permission from the dataset maintainers. You should contact the relevant authorities for access.

### ChestMNIST
- **Source**: [ChestMNIST Dataset](https://www.kaggle.com/datasets/kmader/chest-xray-pneumonia)
- **Description**: A dataset of chest X-ray images. You can download it from Kaggle. Please check Kaggle's terms of use and ensure compliance with their policies.

### Brain Tumor Segmentation (BraTS) Dataset
- **Source**: [BraTS Challenge](https://www.med.upenn.edu/cbica/brats2020/data.html)
- **Description**: A dataset for brain tumor segmentation. It is used in the BraTS challenge, and you can find the dataset and registration information on the provided link.

### PathMNIST
- **Source**: [PathMNIST Dataset](https://zenodo.org/record/12144568)
- **Description**: A histopathology dataset. You can access it through Zenodo, which provides various options for downloading and citing the dataset.

### MIT-BIH Arrhythmia Dataset
- **Source**: [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/)
- **Description**: A database of electrocardiogram (ECG) signals. It is available through PhysioNet. You may need to follow their procedures for accessing and using the data.

### Pediatric MRI Dataset
- **Source**: [Pediatric MRI Dataset Repository]()
- **Description**: (Please provide the actual source if available) If you are using a specific Pediatric MRI dataset, provide the source here. Some Pediatric MRI datasets may have restricted access due to patient privacy concerns, and you may need to follow specific procedures for access.


This README provides an overview of the current state of the project, its components, and future directions. Stay tuned for updates as we continue to develop and improve the federated learning framework.
