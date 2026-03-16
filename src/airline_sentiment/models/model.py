import torch.nn as nn

class SentimentANN(nn.Module):
    """The neural network architecture that performs the classification."""

    def __init__(self, input_dim, hidden_dim=256, output_dim=3):
        super(SentimentANN, self).__init__()

        # Sequence of layers: Linear -> Batch Normalization -> ReLU -> Dropout
        self.layer_stack = nn.Sequential(
            # First Hidden Layer
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Second Hidden Layer
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            # Output Layer (3 neurons for Negative, Neutral, Positive)
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        return self.layer_stack(x)