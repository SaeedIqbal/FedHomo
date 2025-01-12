import torch


class Server:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.clients = []
        self.global_model = None

    def register_client(self, client):
        self.clients.append(client)
        if self.global_model is None:
            self.global_model = client.model.state_dict()

    def federated_train(self, num_rounds=10):
        for round in range(num_rounds):
            local_models = []
            for client in self.clients:
                client.train()
                local_models.append(client.model.state_dict())
            self.global_model = self._aggregate_models(local_models)
            for client in self.clients:
                client.model.load_state_dict(self.global_model)

    def _aggregate_models(self, local_models):
        global_model_state_dict = {}
        for key in local_models[0].keys():
            tensors = [model[key] for model in local_models]
            global_model_state_dict[key] = torch.mean(torch.stack(tensors), dim=0)
        return global_model_state_dict