import torch
import numpy as np


class HierarchicalPrototyping:
    def __init__(self, clients):
        self.clients = clients
        self.global_prototypes = None
        self.local_prototypes = defaultdict(lambda: None)
        self.initialize_prototypes()

    def initialize_prototypes(self):
        # Initialize global prototypes
        all_data = []
        all_targets = []
        for client in self.clients:
            for data, target in client.train_loader:
                all_data.append(data)
                all_targets.append(target)
        all_data = torch.cat(all_data)
        all_targets = torch.cat(all_targets)
        if all_data.dim() == 2:  # Tabular data
            k = min(100, len(all_data))
            indices = np.random.choice(len(all_data), k, replace=False)
            self.global_prototypes = all_data[indices]
        else:  # Image data, assume simple flattening
            all_data_flat = all_data.view(all_data.size(0), -1)
            k = min(100, len(all_data_flat))
            indices = np.random.choice(len(all_data_flat), k, replace=False)
            self.global_prototypes = all_data_flat[indices]

        # Initialize local prototypes
        for client in self.clients:
            client_data = []
            client_targets = []
            for data, target in client.train_loader:
                client_data.append(data)
                client_targets.append(target)
            client_data = torch.cat(client_data)
            client_targets = torch.cat(client_targets)
            if client_data.dim() == 2:  # Tabular data
                k_i = int(np.ceil(np.sqrt(len(client_data))))
                indices = np.random.choice(len(client_data), k_i, replace=False)
                self.local_prototypes[client.client_id] = client_data[indices]
            else:  # Image data, assume simple flattening
                client_data_flat = client_data.view(client_data.size(0), -1)
                k_i = int(np.ceil(np.sqrt(len(client_data_flat))))
                indices = np.random.choice(len(client_data_flat), k_i, replace=False)
                self.local_prototypes[client.client_id] = client_data_flat[indices]

    def calculate_adaptive_weights(self):
        weights = {}
        q_values = []
        for client in self.clients:
            client_data = []
            client_targets = []
            for data, target in client.train_loader:
                client_data.append(data)
                client_targets.append(target)
            client_data = torch.cat(client_data)
            client_targets = torch.cat(client_targets)
            if client_data.dim() == 2:  # Tabular data
                valid_data_points = torch.sum(client_data!= 0, dim=1) > 0
                q_i = valid_data_points.sum().float() / len(client_data)
            else:  # Image data, calculate SNR
                q_i = self.calculate_snr(client_data)
            q_values.append(q_i)
        avg_q = torch.mean(torch.stack(q_values))
        for client in self.clients:
            q_i = q_values[self.clients.index(client)]
            omega_g = q_i / (q_i + avg_q)
            omega_l = 1 - omega_g
            weights[client.client_id] = (omega_g, omega_l)
        return weights

    def update_prototypes(self):
        weights = self.calculate_adaptive_weights()
        new_global_prototypes = []
        for i in range(len(self.global_prototypes)):
            weighted_sum = 0
            total_weight = 0
            for client in self.clients:
                omega_l = weights[client.client_id][1]
                client_prototypes = self.local_prototypes[client.client_id]
                if client_prototypes.dim() == 2:  # Tabular data
                    weighted_sum += omega_l * client_prototypes[i]
                    total_weight += omega_l
                else:  # Image data, assume flattening
                    weighted_sum += omega_l * client_prototypes[i].view(-1)
                    total_weight += omega_l
            if self.global_prototypes.dim() == 2:  # Tabular data
                new_global_prototypes.append(weighted_sum / total_weight)
            else:  # Image data, assume flattening
                new_global_prototypes.append(weighted_sum.view(self.global_prototypes[i].shape) / total_weight)
        self.global_prototypes = torch.stack(new_global_prototypes)

        for client in self.clients:
            omega_g, omega_l = weights[client.client_id]
            new_local_prototypes = []
            for i in range(len(self.local_prototypes[client.client_id])):
                local_prototype = self.local_prototypes[client.client_id][i]
                global_prototype = self.global_prototypes[i]
                if local_prototype.dim() == 2:  # Tabular data
                    # Gradient descent update
                    updated_prototype = self.gradient_descent_update(local_prototype, global_prototype)
                else:  # Image data, assume flattening
                    updated_prototype = self.gradient_descent_update(local_prototype.view(-1), global_prototype.view(-1)).view(local_prototype.shape)
                new_local_prototypes.append(updated_prototype)
            if self.local_prototypes[client.client_id].dim() == 2:
                self.local_prototypes[client.client_id] = torch.stack(new_local_prototypes)
            else:
                self.local_prototypes[client.client_id] = torch.stack(new_local_prototypes).view(self.local_prototypes[client.client_id].shape)

    def enforce_consistency(self, mu=0.1, nu=0.1):
        # Calculate local loss
        local_losses = []
        for client in self.clients:
            local_loss = self.calculate_local_loss(client)
            local_losses.append(local_loss)

        # Calculate hierarchical regularization loss
        hier_losses = []
        for client in self.clients:
            hier_loss = self.calculate_hierarchical_regularization_loss(client)
            hier_losses.append(hier_loss)

        # Calculate distribution consistency loss
        dist_losses = []
        for client in self.clients:
            dist_loss = self.calculate_distribution_consistency_loss(client)
            dist_losses.append(dist_loss)

        for client in self.clients:
            client_index = self.clients.index(client)
            combined_loss = local_losses[client_index] + mu * hier_losses[client_index] + nu * dist_losses[client_index]
            # Update local prototypes using the combined loss
            client.update_prototypes(combined_loss)

    def calculate_snr(self, data):
        # Calculate Signal-to-Noise Ratio (SNR) for image data
        # Assuming data is in the format [batch_size, channels, height, width]
        if data.dim() == 4:
            signal_power = torch.mean(data ** 2)
            noise_power = torch.var(data)
            snr = 10 * torch.log10(signal_power / noise_power)
            return snr
        else:
            raise ValueError("SNR calculation is designed for 4D image data.")

    def gradient_descent_update(self, local_prototype, global_prototype, learning_rate=0.01):
        # Simple gradient descent update
        updated_prototype = local_prototype - learning_rate * (local_prototype - global_prototype)
        return updated_prototype

    def calculate_local_loss(self, client):
        # Calculate local loss based on client's data and prototypes
        # Assuming a simple mean squared error loss for demonstration
        loss = 0
        for i, (data, _) in enumerate(client.train_loader):
            for j in range(len(self.local_prototypes[client.client_id])):
                prototype = self.local_prototypes[client.client_id][j]
                if data.dim() == 2:  # Tabular data
                    # Calculate distance (e.g., Euclidean)
                    distances = torch.norm(data - prototype, dim=1)
                    loss += torch.mean(distances)
                else:  # Image data, assume flattening
                    distances = torch.norm(data.view(data.size(0), -1) - prototype.view(-1), dim=1)
                    loss += torch.mean(distances)
        return loss / (i + 1)

    def calculate_hierarchical_regularization_loss(self, client):
        # Calculate hierarchical regularization loss
        # Assuming a simple mean squared error loss between local and global prototypes
        loss = 0
        for i in range(len(self.local_prototypes[client.client_id])):
            local_prototype = self.local_prototypes[client.client_id][i]
            global_prototype = self.global_prototypes[i]
            if local_prototype.dim() == 2:  # Tabular data
                loss += torch.norm(local_prototype - global_prototype) ** 2
            else:  # Image data, assume flattening
                loss += torch.norm(local_prototype.view(-1) - global_prototype.view(-1)) ** 2
        return loss / len(self.local_prototypes[client.client_id])

    def calculate_distribution_consistency_loss(self, client):
        # Calculate distribution consistency loss
        # Assuming a simple entropy-based loss for demonstration
        loss = 0
        for i, (data, _) in enumerate(client.train_loader):
            if data.dim() == 2:  # Tabular data
                entropy_values = []
                for j in range(data.shape[1]):
                    entropy_values.append(entropy(np.unique(data[:, j], return_counts=True)[1]))
                loss += np.mean(entropy_values)
            else:  # Image data, assume flattening
                flattened_data = data.view(data.size(0), -1)
                entropy_values = []
                for j in range(flattened_data.shape[1]):
                    entropy_values.append(entropy(np.unique(flattened_data[:, j], return_counts=True)[1]))
                loss += np.mean(entropy_values)
        return loss / (i + 1)