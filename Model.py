import torch
import torch.nn as nn
import torch.nn.functional as F

class PCNN_3Branch(nn.Module):
    def __init__(self, input_channels=8, num_classes=2, input_length=125):
        super().__init__()


        self.branch1 = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=20, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2),
        )


        self.branch2 = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=10, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2),
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2),
        )

        self.branch4 = nn.Sequential(
            nn.Conv1d(input_channels, 16, kernel_size=5, padding='same'),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(2),
        )


        with torch.no_grad():
            example_input = torch.randn(1, input_channels, input_length)
            out1 = self.branch1(example_input)
            out2 = self.branch2(example_input)
            out3 = self.branch3(example_input)
            out4 = self.branch4(example_input)
            self.fc_input_dim = out1.numel() + out2.numel() + out3.numel() + out4.numel()

        self.fc = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        combined = torch.cat([out1.flatten(1), out2.flatten(1), out3.flatten(1), out4.flatten(1)], dim=1)
        return self.fc(combined)