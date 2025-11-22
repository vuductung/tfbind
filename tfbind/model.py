import torch
import torch.nn as nn


class DNAConvNet(nn.Module):
    """
    Convolutional Neural Network for DNA sequence classification.

    Architecture:
    - 2 Conv1D blocks (Conv → GELU → MaxPool)
    - Flatten
    - 2 Fully connected layers with dropout
    - Sigmoid output for binary classification

    Args:
        in_channels (int): Number of input channels (4 for DNA: A, C, G, T)
        conv_channels (int): Number of output channels for conv layers
        kernel_size (int): Size of convolutional kernels
        dropout_rate (float): Dropout probability (default: 0.3)
    """

    def __init__(self, in_channels=4, conv_channels=64, kernel_size=10, dropout_rate=0.2):
        super().__init__()

        # Feature extraction with convolutional layers
        self.feature_extractor = nn.Sequential(
            # First conv block
            nn.Conv1d(in_channels=in_channels, out_channels=conv_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
            # Second conv block
            nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=kernel_size, padding="same"),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate),
        )

        # Flatten layer (cleaner than manual reshape)
        self.flatten = nn.Flatten()

        # Classification head (initialized dynamically in forward pass)
        self.classifier = None
        self.dropout_rate = dropout_rate

    def _init_classifier(self, flattened_size):
        """Initialize classifier layers based on flattened feature size."""
        self.classifier = nn.Sequential(
            nn.Linear(flattened_size, flattened_size // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(flattened_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_length)

        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid probabilities
        """
        # Extract features
        x = self.feature_extractor(x)

        # Flatten
        x = self.flatten(x)

        # Initialize classifier on first forward pass (lazy initialization)
        if self.classifier is None:
            self._init_classifier(x.shape[1])
            # Move classifier to same device as input
            self.classifier = self.classifier.to(x.device)

        # Classification
        x = self.classifier(x)

        return x
