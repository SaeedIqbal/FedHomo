from data_loading import DataHandler
from client import Client
from server import Server
from hierarchical_prototyping import HierarchicalPrototyping


if __name__ == "__main__":
    # Load data
    data_handler = DataHandler('UCI Heart Disease')
    data_handler.prepare_data(batch_size=32)

    # Create clients
    client1 = Client(1, data_handler.train_loader, data_handler.val_loader, data_handler.test_loader)
    client2 = Client(2, data_handler.train_loader, data_handler.val_loader, data_handler.test_loader)
    clients = [client1, client2]

    # Server
    server = Server(len(clients))
    for client in clients:
        server.register_client(client)

    # Hierarchical prototyping
    hp = HierarchicalPrototyping(clients)
    hp.update_prototypes()
    hp.enforce_consistency()