"""
TAWSEEM MLP Model (PyTorch)
Multilayer Perceptron for DNA contributor number estimation.
"""

import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import NUM_HIDDEN_LAYERS, HIDDEN_DIM, DROPOUT_RATE, NUM_CLASSES


class TAWSEEM_MLP(nn.Module):
    """MLP for profile-level NOC prediction (flat feature vector)."""
    
    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES,
                 dropout_rate=DROPOUT_RATE):
        super(TAWSEEM_MLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(32, num_classes),
        )
        
        self._input_dim = input_dim
        self._num_classes = num_classes
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)
    
    def summary(self):
        linear_layers = [l for l in self.network if isinstance(l, nn.Linear)]
        print(f"TAWSEEM MLP Model:")
        print(f"  Input dim:     {self._input_dim}")
        print(f"  Architecture:  {' -> '.join([str(self._input_dim)] + [str(l.out_features) for l in linear_layers])}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total params:  {total_params:,}")


class TAWSEEM_CNN(nn.Module):
    """
    1D CNN for profile-level NOC prediction.
    
    Treats each profile as a sequence of markers:
    Input: (batch, n_features_per_marker, n_markers) — Conv1d channels-first
    
    Conv layers learn patterns across neighboring markers.
    Global average pooling aggregates all marker info.
    """
    
    def __init__(self, n_features_per_marker=11, n_markers=22, num_classes=NUM_CLASSES):
        super(TAWSEEM_CNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Block 1
            nn.Conv1d(n_features_per_marker, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Block 2
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Block 3
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )
        
        self._n_features = n_features_per_marker
        self._n_markers = n_markers
        self._num_classes = num_classes
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x: (batch, n_features_per_marker, n_markers)
        h = self.conv_layers(x)          # (batch, 128, n_markers)
        h = h.mean(dim=2)                # Global average pooling → (batch, 128)
        return self.classifier(h)        # (batch, num_classes)
    
    def summary(self):
        print(f"TAWSEEM 1D-CNN Model:")
        print(f"  Input: ({self._n_features}, {self._n_markers})")
        print(f"  Conv:  {self._n_features} -> 64 -> 128 -> 128 (kernel=3)")
        print(f"  Pool:  GlobalAvgPool -> 128")
        print(f"  FC:    128 -> 64 -> {self._num_classes}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"  Total params: {total_params:,}")
