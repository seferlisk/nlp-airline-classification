import torch
import torch.nn as nn
import torch.optim as optim

class ModelTrainer:
    """Handles the training loop, Early Stopping, and Loss monitoring."""

    def __init__(self, model, lr=0.001, patience=5):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.patience = patience

    def train_step(self, train_loader, val_loader, epochs=50):
        train_losses, val_losses = [], []
        best_loss = float('inf')
        counter = 0

        for epoch in range(epochs):
            self.model.train()
            t_loss = 0
            for X, y in train_loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(X), y)
                loss.backward();
                self.optimizer.step()
                t_loss += loss.item()

            self.model.eval()
            v_loss = 0
            with torch.no_grad():
                for X, y in val_loader:
                    v_loss += self.criterion(self.model(X), y).item()

            avg_t, avg_v = t_loss / len(train_loader), v_loss / len(val_loader)
            train_losses.append(avg_t);
            val_losses.append(avg_v)

            if avg_v < best_loss:
                best_loss = avg_v
                counter = 0
            else:
                counter += 1
                if counter >= self.patience: break
        return train_losses, val_losses