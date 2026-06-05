import torch.nn as nn


class DeepLightweightMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=20, dropout_rate=0.2):
        super(DeepLightweightMLP, self).__init__()

        # 1. Input Projection: Bring 768 down to a lightweight hidden size
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 2. Deep Hidden Core: Deep processing at a fixed, low-parameter width
        # Using GELU activation for smoother gradients in deeper networks
        self.hidden_block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.hidden_block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        self.hidden_block3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )

        # 3. Output Classifier: Map the final hidden states to the 20 classes
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def features(self, x):
        # Extract features from the input (useful for feature-based methods)
        x = self.input_layer(x)
        x = self.hidden_block1(x)
        x = self.hidden_block2(x)
        x = self.hidden_block3(x)
        return x  # Return the final hidden representation before classification

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
