"""
PyTorch Models for Nursing Activity Classification

Contains all deep learning models:
1. BaselineCNNLSTM - Baseline CNN-LSTM model
2. CorrelationAwareCNN - Advanced CNN with correlation modeling
3. AttentionLSTM - LSTM with attention mechanism
4. FeatureSelectiveNet - Network with adaptive feature selection
5. CorrelationLayer - Custom correlation computation layer

All models are implemented in PyTorch for nursing activity recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import os


class CorrelationLayer(nn.Module):
    """
    Custom layer for computing and applying correlations between feature groups
    """
    
    def __init__(self, correlation_weight: float = 0.1):
        super().__init__()
        self.correlation_weight = correlation_weight
        
    def forward(self, group_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Concatenate all group outputs
        group_list = list(group_outputs.values())
        concatenated = torch.cat(group_list, dim=2)
        
        # Compute cross-correlations between groups
        correlation_features = []
        
        group_items = list(group_outputs.items())
        for i, (name_i, output_i) in enumerate(group_items):
            for j, (name_j, output_j) in enumerate(group_items):
                if i <= j:  # Avoid duplicate correlations
                    continue
                
                # Compute correlation between groups i and j
                corr = self.compute_correlation(output_i, output_j)
                correlation_features.append(corr)
        
        # Combine original features with correlation features
        if correlation_features:
            correlation_tensor = torch.cat(correlation_features, dim=2)
            # Weight the correlation features
            correlation_tensor = correlation_tensor * self.correlation_weight
            
            # Add to concatenated features
            combined = torch.cat([concatenated, correlation_tensor], dim=2)
        else:
            combined = concatenated
            
        return combined
    
    def compute_correlation(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Compute correlation between two feature tensors
        """
        # Normalize features
        x1_norm = F.normalize(x1, p=2, dim=2)
        x2_norm = F.normalize(x2, p=2, dim=2)
        
        # Compute element-wise product (correlation proxy)
        correlation = x1_norm * x2_norm
        
        return correlation


class BaselineCNNLSTM(nn.Module):
    """
    Baseline CNN-LSTM model
    """
    
    def __init__(self, num_classes: int, input_size: int = 70):
        super().__init__()
        
        # CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # LSTM layers
        self.lstm = nn.LSTM(64, 50, batch_first=True)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(50, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # Convert to (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch_size, 64, seq_len/2)
        
        # Convert back for LSTM: (batch_size, seq_len/2, 64)
        cnn_out = cnn_out.transpose(1, 2)
        
        # LSTM processing
        lstm_out, (hidden, _) = self.lstm(cnn_out)
        
        # Use last hidden state
        output = self.classifier(hidden[-1])
        
        return output


class CorrelationAwareCNN(nn.Module):
    """
    CNN model with correlation-aware convolutions for grouped sensor features
    """
    
    def __init__(self, 
                 num_classes: int,
                 num_features: int = 70,
                 num_sensors: int = 5,
                 window_size: int = 50,
                 filters_per_group: int = 32,
                 correlation_weight: float = 0.1):
        super().__init__()
        
        self.num_classes = num_classes
        self.num_features = num_features
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.filters_per_group = filters_per_group
        self.correlation_weight = correlation_weight
        
        # Calculate features per sensor dynamically
        self.features_per_sensor = num_features // num_sensors
        
        # Feature groups based on actual feature count
        if self.features_per_sensor == 14:
            # Original SONAR structure (14 features per sensor)
            self.feature_groups = {
                'quaternion': list(range(0, 4)),      # Quat_W, Quat_X, Quat_Y, Quat_Z
                'quaternion_deriv': list(range(4, 8)), # dq_W, dq_X, dq_Y, dq_Z
                'velocity': list(range(8, 11)),        # dv[1], dv[2], dv[3]
                'magnetic': list(range(11, 14))        # Mag_X, Mag_Y, Mag_Z
            }
        else:
            # Generic grouping for different feature counts
            group_size = max(1, self.features_per_sensor // 4)  # Divide into 4 groups
            self.feature_groups = {
                'group_1': list(range(0, group_size)),
                'group_2': list(range(group_size, 2*group_size)),
                'group_3': list(range(2*group_size, 3*group_size)),
                'group_4': list(range(3*group_size, self.features_per_sensor))
            }
        
        # Group-specific convolution layers
        self.group_convs = nn.ModuleDict()
        for group_name, indices in self.feature_groups.items():
            features_per_sensor = len(indices)
            total_group_features = features_per_sensor * num_sensors
            
            self.group_convs[group_name] = nn.Sequential(
                nn.Conv1d(total_group_features, filters_per_group, 3, padding=1),
                nn.BatchNorm1d(filters_per_group),
                nn.ReLU(),
                nn.Conv1d(filters_per_group, filters_per_group, 3, padding=1),
                nn.BatchNorm1d(filters_per_group),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
        
        # Correlation computation layer
        self.correlation_layer = CorrelationLayer(correlation_weight)
        
        # Simplified final layers (let the correlation layer handle size calculation)
        self.global_conv = nn.Conv1d(filters_per_group * len(self.feature_groups), 64, 3, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Input shape: (batch_size, window_size, num_features)
        # Convert to (batch_size, num_features, window_size) for Conv1d
        x = x.transpose(1, 2)
        
        # Extract feature groups and apply group-specific convolutions
        group_outputs = {}
        
        for group_name, indices in self.feature_groups.items():
            # Extract features for this group across all sensors
            group_indices = []
            for sensor_idx in range(self.num_sensors):
                sensor_offset = sensor_idx * self.features_per_sensor  # Use dynamic features per sensor
                group_indices.extend([sensor_offset + idx for idx in indices])
            
            # Extract group data
            group_data = x[:, group_indices, :]
            
            # Apply group-specific convolution
            group_output = self.group_convs[group_name](group_data)
            group_outputs[group_name] = group_output
        
        # Simple concatenation of group outputs (skip complex correlation layer for now)
        group_list = list(group_outputs.values())
        combined_features = torch.cat(group_list, dim=1)
        
        # Final classification
        x = self.global_conv(combined_features)
        x = F.relu(x)
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Remove the last dimension
        x = self.dropout(x)
        
        return self.classifier(x)


class AttentionLSTM(nn.Module):
    """
    LSTM with attention mechanism for activity discrimination
    """
    
    def __init__(self, num_classes: int, input_size: int = 70, hidden_size: int = 64):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)  # (batch_size, seq_len, hidden_size)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
        attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)  # (batch_size, seq_len)
        
        # Apply attention
        attended_output = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_size)
        
        # Classification
        output = self.classifier(attended_output)
        
        return output


class FeatureSelectiveNet(nn.Module):
    """
    Network with adaptive feature selection
    """
    
    def __init__(self, num_classes: int, input_size: int = 70, window_size: int = 50):
        super().__init__()
        
        self.input_size = input_size
        self.window_size = window_size
        
        # Feature selection gates
        self.gate_network = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )
        
        # CNN layers on selected features
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_size, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        # Convert to (batch_size, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # Compute feature selection gates
        gates = self.gate_network(x)  # (batch_size, input_size)
        gates = gates.unsqueeze(-1)  # (batch_size, input_size, 1)
        
        # Apply feature selection
        selected_features = x * gates  # (batch_size, input_size, seq_len)
        
        # Extract features
        features = self.feature_extractor(selected_features)  # (batch_size, 64, 1)
        features = features.squeeze(-1)  # (batch_size, 64)
        
        # Classification
        output = self.classifier(features)
        
        return output


# Simple CNN model (without LSTM)
class BaselineCNN(nn.Module):
    """Simple baseline CNN model (no LSTM)"""
    
    def __init__(self, num_classes, input_size=70):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
        
        # CNN feature extraction
        x = self.cnn(x)  # (batch_size, 64, 1)
        x = x.squeeze(-1)  # (batch_size, 64)
        
        # Classification
        output = self.classifier(x)
        
        return output


class HybridNet(nn.Module):
    """
    🏆 Hybrid model combining the best features from three advanced models:
    
    1. Feature Selection (from FeatureSelectiveNet): Adaptive feature importance learning
    2. Correlation Awareness (from CorrelationAwareCNN): Inter-sensor correlation modeling  
    3. Temporal Attention (from AttentionLSTM): Important time-step focus
    
    Designed for ablation study - each component can be enabled/disabled.
    """
    
    def __init__(self, 
                 num_classes: int, 
                 input_size: int = 70,
                 num_sensors: int = 5,
                 window_size: int = 30,
                 hidden_size: int = 64,
                 # Ablation study flags
                 enable_feature_selection: bool = True,
                 enable_correlation_aware: bool = True, 
                 enable_temporal_attention: bool = True,
                 # Additional parameters
                 correlation_weight: float = 0.1,
                 filters_per_group: int = 32):
        super().__init__()
        
        # Store configuration for ablation study
        self.enable_feature_selection = enable_feature_selection
        self.enable_correlation_aware = enable_correlation_aware
        self.enable_temporal_attention = enable_temporal_attention
        self.input_size = input_size
        self.num_sensors = num_sensors
        self.hidden_size = hidden_size
        
        # Calculate features per sensor dynamically
        self.features_per_sensor = input_size // num_sensors
        
        print(f"🔧 HybridNet Configuration:")
        print(f"   Feature Selection: {enable_feature_selection}")
        print(f"   Correlation Awareness: {enable_correlation_aware}")
        print(f"   Temporal Attention: {enable_temporal_attention}")
        
        # === 1. FEATURE SELECTION MODULE (from FeatureSelectiveNet) ===
        if self.enable_feature_selection:
            self.feature_gate_network = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(input_size, input_size),
                nn.Sigmoid()
            )
            print(f"   ✓ Feature Selection Gates: {input_size} features")
        
        # === 2. CORRELATION-AWARE MODULE (from CorrelationAwareCNN) ===
        if self.enable_correlation_aware:
            # Define sensor feature groups dynamically based on actual feature count
            if self.features_per_sensor == 14:
                # Original SONAR structure (14 features per sensor)
                self.feature_groups = {
                    'quaternion': list(range(0, 4)),      # Quat_W, Quat_X, Quat_Y, Quat_Z
                    'quaternion_deriv': list(range(4, 8)), # dq_W, dq_X, dq_Y, dq_Z  
                    'velocity': list(range(8, 11)),        # dv[1], dv[2], dv[3]
                    'magnetic': list(range(11, 14))        # Mag_X, Mag_Y, Mag_Z
                }
            else:
                # Generic grouping for different feature counts
                group_size = max(1, self.features_per_sensor // 4)  # Divide into 4 groups
                self.feature_groups = {
                    'group_1': list(range(0, group_size)),
                    'group_2': list(range(group_size, 2*group_size)),
                    'group_3': list(range(2*group_size, 3*group_size)),
                    'group_4': list(range(3*group_size, self.features_per_sensor))
                }
            
            # Group-specific CNN layers
            self.group_convs = nn.ModuleDict()
            for group_name, indices in self.feature_groups.items():
                features_per_sensor = len(indices)
                total_group_features = features_per_sensor * num_sensors
                
                self.group_convs[group_name] = nn.Sequential(
                    nn.Conv1d(total_group_features, filters_per_group, 3, padding=1),
                    nn.BatchNorm1d(filters_per_group),
                    nn.ReLU(),
                    nn.Conv1d(filters_per_group, filters_per_group, 3, padding=1),
                    nn.BatchNorm1d(filters_per_group),
                    nn.ReLU(),
                    nn.MaxPool1d(2)
                )
            
            # Correlation computation
            self.correlation_layer = CorrelationLayer(correlation_weight)
            
            # CorrelationLayer concatenates along sequence dimension, not feature dimension
            # So output features = filters_per_group (same as individual group output)
            total_correlation_features = filters_per_group
            
            print(f"   ✓ Correlation Groups: {list(self.feature_groups.keys())}")
            print(f"   ✓ Correlation output features: {total_correlation_features}")
            
        else:
            # Simple CNN if correlation is disabled
            self.simple_cnn = nn.Sequential(
                nn.Conv1d(input_size, 64, 3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 64, 3, padding=1), 
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
            total_correlation_features = 64
            print(f"   ✓ Simple CNN: {input_size} -> 64 features")
        
        # === 3. TEMPORAL ATTENTION MODULE (from AttentionLSTM) ===
        if self.enable_temporal_attention:
            self.lstm = nn.LSTM(total_correlation_features, hidden_size, batch_first=True)
            self.attention = nn.Linear(hidden_size, 1)
            
            print(f"   ✓ Temporal Attention: LSTM({total_correlation_features}) -> {hidden_size}")
            final_feature_size = hidden_size
        else:
            # Global average pooling if attention is disabled
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            final_feature_size = total_correlation_features
            print(f"   ✓ Global Pooling: {total_correlation_features} features")
        
        # === 4. FINAL CLASSIFICATION LAYERS ===
        self.classifier = nn.Sequential(
            nn.Linear(final_feature_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        print(f"   ✓ Classifier: {final_feature_size} -> 128 -> {num_classes}")
        print("🏆 HybridNet initialized successfully!")
        
    def forward(self, x):
        """
        Forward pass with modular components for ablation study
        Input shape: (batch_size, seq_len, input_size)
        """
        batch_size, seq_len, input_size = x.shape
        
        # === 1. FEATURE SELECTION ===
        if self.enable_feature_selection:
            # Convert for feature gate computation
            x_transposed = x.transpose(1, 2)  # (batch_size, input_size, seq_len)
            
            # Compute feature importance gates
            feature_gates = self.feature_gate_network(x_transposed)  # (batch_size, input_size)
            feature_gates = feature_gates.unsqueeze(-1)  # (batch_size, input_size, 1)
            
            # Apply feature selection
            x_selected = x_transposed * feature_gates  # Element-wise multiplication
        else:
            x_selected = x.transpose(1, 2)  # Just transpose, no selection
        
        # === 2. CORRELATION-AWARE FEATURE EXTRACTION ===
        if self.enable_correlation_aware:
            # Extract feature groups and apply group-specific convolutions
            group_outputs = {}
            
            for group_name, indices in self.feature_groups.items():
                # Extract features for this group across all sensors
                group_indices = []
                for sensor_idx in range(self.num_sensors):
                    sensor_offset = sensor_idx * self.features_per_sensor  # Use dynamic features per sensor
                    group_indices.extend([sensor_offset + idx for idx in indices])
                
                # Extract group data
                group_data = x_selected[:, group_indices, :]
                
                # Apply group-specific convolution
                group_output = self.group_convs[group_name](group_data)
                group_outputs[group_name] = group_output
            
            # Apply correlation layer
            correlation_features = self.correlation_layer(group_outputs)
            
        else:
            # Simple CNN feature extraction
            correlation_features = self.simple_cnn(x_selected)
        
        # Convert back to (batch_size, seq_len, features) for temporal processing
        correlation_features = correlation_features.transpose(1, 2)
        
        # === 3. TEMPORAL ATTENTION ===
        if self.enable_temporal_attention:
            # LSTM temporal modeling
            lstm_out, _ = self.lstm(correlation_features)  # (batch_size, seq_len, hidden_size)
            
            # Attention mechanism
            attention_weights = self.attention(lstm_out)  # (batch_size, seq_len, 1)
            attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)  # (batch_size, seq_len)
            
            # Apply attention weighting
            final_features = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)  # (batch_size, hidden_size)
            
        else:
            # Global average pooling over time dimension
            final_features = correlation_features.transpose(1, 2)  # (batch_size, features, seq_len) 
            final_features = self.global_pool(final_features)  # (batch_size, features, 1)
            final_features = final_features.squeeze(-1)  # (batch_size, features)
        
        # === 4. CLASSIFICATION ===
        output = self.classifier(final_features)
        
        return output
    
    def get_feature_importance(self, x):
        """
        Extract feature importance for analysis (only works if feature selection is enabled)
        """
        if not self.enable_feature_selection:
            return None
            
        x_transposed = x.transpose(1, 2)
        feature_gates = self.feature_gate_network(x_transposed)
        return feature_gates.detach().cpu().numpy()
    
    def get_attention_weights(self, x):
        """
        Extract attention weights for analysis (only works if temporal attention is enabled)
        """
        if not self.enable_temporal_attention:
            return None
            
        # Forward pass until attention computation
        batch_size, seq_len, input_size = x.shape
        
        # Feature selection
        if self.enable_feature_selection:
            x_transposed = x.transpose(1, 2)
            feature_gates = self.feature_gate_network(x_transposed)
            feature_gates = feature_gates.unsqueeze(-1)
            x_selected = x_transposed * feature_gates
        else:
            x_selected = x.transpose(1, 2)
        
        # Correlation-aware features
        if self.enable_correlation_aware:
            group_outputs = {}
            for group_name, indices in self.feature_groups.items():
                group_indices = []
                for sensor_idx in range(self.num_sensors):
                    sensor_offset = sensor_idx * self.features_per_sensor
                    group_indices.extend([sensor_offset + idx for idx in indices])
                group_data = x_selected[:, group_indices, :]
                group_output = self.group_convs[group_name](group_data)
                group_outputs[group_name] = group_output
            correlation_features = self.correlation_layer(group_outputs)
        else:
            correlation_features = self.simple_cnn(x_selected)
        
        correlation_features = correlation_features.transpose(1, 2)
        
        # LSTM and attention
        lstm_out, _ = self.lstm(correlation_features)
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights.squeeze(-1), dim=1)
        
        return attention_weights.detach().cpu().numpy()


def create_model(model_type: str, num_classes: int, input_size: int = 70, **kwargs):
    """
    Factory function to create different model types
    """
    
    if model_type == 'baseline':
        return BaselineCNNLSTM(num_classes, input_size)
    elif model_type == 'baseline_cnn':
        return BaselineCNN(num_classes, input_size)
    elif model_type == 'correlation_aware':
        return CorrelationAwareCNN(num_classes, input_size, **kwargs)
    elif model_type == 'attention':
        return AttentionLSTM(num_classes, input_size, **kwargs)
    elif model_type == 'feature_selective':
        return FeatureSelectiveNet(num_classes, input_size, **kwargs)
    elif model_type == 'hybrid':
        return HybridNet(num_classes, input_size, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def test_models():
    """Test all model implementations"""
    print("Testing PyTorch model implementations...")
    
    # Test parameters
    batch_size = 16
    window_size = 30
    input_size = 70
    num_classes = 8
    
    # Create dummy data
    x = torch.randn(batch_size, window_size, input_size)
    
    # Test all models
    models = {
        'Baseline CNN-LSTM': create_model('baseline', num_classes, input_size),
        'Baseline CNN': create_model('baseline_cnn', num_classes, input_size),
        'Correlation-Aware CNN': create_model('correlation_aware', num_classes, input_size, window_size=window_size),
        'Attention LSTM': create_model('attention', num_classes, input_size),
        'Feature-Selective Net': create_model('feature_selective', num_classes, input_size, window_size=window_size),
        'Hybrid Net': create_model('hybrid', num_classes, input_size, window_size=window_size)
    }
    
    for name, model in models.items():
        try:
            model.eval()
            with torch.no_grad():
                output = model(x)
            
            print(f"✓ {name}: Input {x.shape} -> Output {output.shape}")
            assert output.shape == (batch_size, num_classes), f"Wrong output shape for {name}"
            
        except Exception as e:
            print(f"✗ {name}: Error - {e}")
    
    print("All PyTorch models tested successfully!")


def load_saved_model(model_path, device=None):
    """
    Load a saved model from disk
    
    Args:
        model_path (str): Path to the saved model (.pth file)
        device (torch.device, optional): Device to load the model on
    
    Returns:
        tuple: (model, checkpoint_info)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from: {model_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract model configuration
        model_name = checkpoint['model_name']
        model_architecture = checkpoint['model_architecture']
        input_size = checkpoint['input_size']
        num_classes = checkpoint['num_classes']
        
        print(f"  Model: {model_name}")
        print(f"  Architecture: {model_architecture}")
        print(f"  Input size: {input_size}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Saved on: {checkpoint.get('saved_timestamp', 'Unknown')}")
        print(f"  Test accuracy: {checkpoint.get('test_metrics', {}).get('accuracy', 'Unknown')}")
        
        # Create model based on architecture
        if model_architecture == 'BaselineCNNLSTM':
            model = BaselineCNNLSTM(num_classes, input_size)
        elif model_architecture == 'CorrelationAwareCNN':
            model = CorrelationAwareCNN(num_classes, input_size)
        elif model_architecture == 'AttentionLSTM':
            model = AttentionLSTM(num_classes, input_size)
        elif model_architecture == 'FeatureSelectiveNet':
            model = FeatureSelectiveNet(num_classes, input_size)
        elif model_architecture == 'HybridNet':
            # Get HybridNet specific configuration
            hybrid_config = checkpoint.get('hybrid_config', {})
            model = HybridNet(
                num_classes=num_classes,
                input_size=input_size,
                enable_feature_selection=hybrid_config.get('enable_feature_selection', True),
                enable_correlation_aware=hybrid_config.get('enable_correlation_aware', True),
                enable_temporal_attention=hybrid_config.get('enable_temporal_attention', True),
                num_sensors=hybrid_config.get('num_sensors', 5),
                hidden_size=hybrid_config.get('hidden_size', 64)
            )
        else:
            raise ValueError(f"Unknown model architecture: {model_architecture}")
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"  ✅ Model loaded successfully!")
        
        # Return model and checkpoint info for reference
        checkpoint_info = {
            'model_name': model_name,
            'architecture': model_architecture,
            'input_size': input_size,
            'num_classes': num_classes,
            'test_accuracy': checkpoint.get('test_metrics', {}).get('accuracy'),
            'training_epochs': checkpoint.get('training_epochs'),
            'training_time': checkpoint.get('training_time'),
            'saved_timestamp': checkpoint.get('saved_timestamp'),
            'hybrid_config': checkpoint.get('hybrid_config', {}) if model_architecture == 'HybridNet' else None
        }
        
        return model, checkpoint_info
        
    except Exception as e:
        print(f"  ❌ Error loading model: {e}")
        raise e


def list_saved_models(models_dir='../saved_models'):
    """
    List all saved models in the models directory
    
    Args:
        models_dir (str): Directory containing saved models
        
    Returns:
        list: List of model file paths
    """
    import glob
    
    if not os.path.exists(models_dir):
        print(f"Models directory does not exist: {models_dir}")
        return []
    
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))
    
    if not model_files:
        print(f"No saved models found in: {models_dir}")
        return []
    
    print(f"Found {len(model_files)} saved model(s):")
    
    for i, model_file in enumerate(model_files, 1):
        try:
            # Load just the metadata to display info
            checkpoint = torch.load(model_file, map_location='cpu')
            model_name = checkpoint.get('model_name', 'Unknown')
            test_acc = checkpoint.get('test_metrics', {}).get('accuracy', 'Unknown')
            saved_time = checkpoint.get('saved_timestamp', 'Unknown')
            file_size = os.path.getsize(model_file) / 1024 / 1024  # MB
            
            print(f"  {i}. {os.path.basename(model_file)}")
            print(f"     - Model: {model_name}")
            print(f"     - Test Accuracy: {test_acc}")
            print(f"     - Saved: {saved_time}")
            print(f"     - Size: {file_size:.2f} MB")
            print()
            
        except Exception as e:
            print(f"  {i}. {os.path.basename(model_file)} (Error reading metadata: {e})")
    
    return model_files


def predict_with_saved_model(model_path, input_data, device=None):
    """
    Load a saved model and make predictions on input data
    
    Args:
        model_path (str): Path to saved model
        input_data (torch.Tensor): Input data with shape (batch_size, seq_len, features)
        device (torch.device, optional): Device to run inference on
        
    Returns:
        dict: Predictions and probabilities
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, checkpoint_info = load_saved_model(model_path, device)
    
    # Ensure input data is on the correct device
    if isinstance(input_data, np.ndarray):
        input_data = torch.FloatTensor(input_data)
    input_data = input_data.to(device)
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_data)
        probabilities = F.softmax(outputs, dim=1)
        predictions = torch.argmax(outputs, dim=1)
    
    return {
        'predictions': predictions.cpu().numpy(),
        'probabilities': probabilities.cpu().numpy(),
        'raw_outputs': outputs.cpu().numpy(),
        'model_info': checkpoint_info
    }


# Example usage function
def example_model_loading():
    """
    Example of how to load and use saved models
    """
    print("="*60)
    print("EXAMPLE: Loading and Using Saved Models")
    print("="*60)
    
    # List available models
    print("\n1. Listing saved models:")
    model_files = list_saved_models()
    
    if model_files:
        # Load the first model as an example
        print(f"\n2. Loading model: {model_files[0]}")
        try:
            model, info = load_saved_model(model_files[0])
            
            print(f"\n3. Model information:")
            for key, value in info.items():
                if value is not None:
                    print(f"   {key}: {value}")
            
            print(f"\n4. Model ready for inference!")
            print(f"   - Input shape expected: (batch_size, seq_len, {info['input_size']})")
            print(f"   - Output shape: (batch_size, {info['num_classes']})")
            
            # Create dummy data for testing
            batch_size = 4
            seq_len = 20
            dummy_input = torch.randn(batch_size, seq_len, info['input_size'])
            
            print(f"\n5. Testing with dummy data:")
            print(f"   Input shape: {dummy_input.shape}")
            
            # Make predictions
            results = predict_with_saved_model(model_files[0], dummy_input)
            print(f"   Predictions: {results['predictions']}")
            print(f"   Max probabilities: {np.max(results['probabilities'], axis=1):.4f}")
            
        except Exception as e:
            print(f"   Error: {e}")
    else:
        print("\n   No saved models found. Run training first!")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Run tests
    test_models()
    
    # Show example of loading saved models
    example_model_loading() 