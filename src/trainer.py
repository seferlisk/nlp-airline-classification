import torch
import torch.nn as nn
import torch.optim as optim
import os

class ModelTrainer:
    """Handles the training loop, Early Stopping, Loss monitoring and ensures the best model state is preserved."""

    def __init__(self, model, lr=0.001, patience=5, output_dir='Outputs'):
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.patience = patience
        self.output_dir = output_dir  # The folder name
        self.best_val_loss = float('inf')
        self.counter = 0

        # Ensure the Outputs folder exists immediately
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created directory: {self.output_dir}")

    def train(self, train_loader, val_loader, epochs=50):
        """Iterates through epochs, calculating loss and updating weights via the Adam optimizer."""
        train_losses, val_losses = [], []

        # Define the full path for the .pth file
        save_path = os.path.join(self.output_dir, 'best_model_weights.pth')

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            for batch_X, batch_y in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

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

            # EARLY STOPPING & SAVING TO THE OUTPUTS FOLDER
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.counter = 0
                # Save the model to Outputs/best_model_weights.pth
                torch.save(self.model.state_dict(), save_path)
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"--- Early stopping triggered. Best model saved in {self.output_dir} ---")
                    break

        return train_losses, val_losses