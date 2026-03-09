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
        self.best_val_loss = float('inf')
        self.counter = 0

    def train(self, train_loader, val_loader, epochs=50):
        train_losses, val_losses = [], []

        for epoch in range(epochs):
            # --- TRAINING PHASE ---
            self.model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            # --- VALIDATION PHASE ---
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for v_batch_X, v_batch_y in val_loader:
                    v_outputs = self.model(v_batch_X)
                    v_loss = self.criterion(v_outputs, v_batch_y)
                    total_val_loss += v_loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            avg_val_loss = total_val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch + 1:02d}: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # --- EARLY STOPPING LOGIC ---
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.counter = 0
                torch.save(self.model.state_dict(), 'best_model_weights.pth')
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"--- Early stopping triggered after {epoch + 1} epochs ---")
                    break

        return train_losses, val_losses