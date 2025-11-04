"""
Real Data Loader for SONAR Nursing Activity Dataset
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import glob
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class SONARDataLoader:
    """Real data loader for SONAR dataset with correlation analysis"""
    
    def __init__(self, data_dir: str = "../SONAR_ML"):
        self.data_dir = data_dir
        self.sensor_names = ['LF', 'LW', 'ST', 'RW', 'RF']  # Left Foot, Left Wrist, Sternum, Right Wrist, Right Foot
        self.feature_names = [
            'Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z',    # Quaternion orientation
            'dq_W', 'dq_X', 'dq_Y', 'dq_Z',            # Quaternion derivatives
            'dv[1]', 'dv[2]', 'dv[3]',                 # Velocity derivatives (x,y,z)
            'Mag_X', 'Mag_Y', 'Mag_Z'                  # Magnetic field (x,y,z)
        ]
        
        # Feature groups for correlation analysis
        self.feature_groups = {
            'quaternion': [0, 1, 2, 3],           # Quat_W, Quat_X, Quat_Y, Quat_Z
            'quaternion_deriv': [4, 5, 6, 7],     # dq_W, dq_X, dq_Y, dq_Z
            'velocity': [8, 9, 10],               # dv[1], dv[2], dv[3]
            'magnetic': [11, 12, 13]              # Mag_X, Mag_Y, Mag_Z
        }
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.raw_data = None
        self.processed_data = None
        
    def load_raw_data(self, max_files: int = 50) -> pd.DataFrame:
        """Load CSV files from SONAR dataset"""
        print(f"Loading SONAR dataset from {self.data_dir}...")
        
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        all_data = []
        loaded_files = 0
        
        for file_path in csv_files[:max_files]:  # Limit files for initial testing
            try:
                print(f"Loading {os.path.basename(file_path)}...")
                filename = os.path.basename(file_path)
                recording_id = filename.split('_')[0]
                subject_id = filename.split('_')[1].replace('.csv', '')
                
                # Load CSV data
                df = pd.read_csv(file_path)
                
                # Check if required columns exist
                if 'activity' not in df.columns:
                    print(f"Warning: No 'activity' column in {file_path}")
                    continue
                
                # Add metadata
                df['recording_id'] = recording_id
                df['subject_id'] = subject_id
                
                all_data.append(df)
                loaded_files += 1
                
                if loaded_files >= max_files:
                    break
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid data files could be loaded")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        print(f"Successfully loaded {len(combined_data)} samples from {loaded_files} files")
        print(f"Unique activities: {combined_data['activity'].nunique()}")
        print(f"Unique subjects: {combined_data['subject_id'].nunique()}")
        
        # Display basic statistics
        print("\nActivity distribution:")
        activity_counts = combined_data['activity'].value_counts()
        for activity, count in activity_counts.head(10).items():
            print(f"  {activity}: {count} samples")
        
        self.raw_data = combined_data
        return combined_data
    
    def preprocess_data(self, normalize: bool = True, remove_outliers: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Real preprocessing with outlier removal and normalization"""
        if self.raw_data is None:
            raise ValueError("No data loaded. Call load_raw_data() first.")
        
        print("Preprocessing data...")
        
        # Get feature columns (exclude metadata)
        feature_cols = [col for col in self.raw_data.columns 
                       if col not in ['activity', 'recording_id', 'subject_id', 'file_path', 'SampleTimeFine']]
        
        print(f"Found {len(feature_cols)} feature columns")
        
        # Extract features and labels
        X = self.raw_data[feature_cols].copy()
        y = self.raw_data['activity'].copy()
        subjects = self.raw_data['subject_id'].copy()
        
        print(f"Original data shape: {X.shape}")
        print(f"Missing values: {X.isnull().sum().sum()}")
        
        # Handle missing values
        X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # Remove rows with any remaining NaN values
        valid_indices = ~X.isnull().any(axis=1) & ~y.isnull()
        X = X[valid_indices]
        y = y[valid_indices]
        subjects = subjects[valid_indices]
        
        print(f"After cleaning: {X.shape}")
        
        # Remove outliers if requested
        if remove_outliers:
            print("Removing outliers...")
            # Use IQR method for outlier detection
            Q1 = X.quantile(0.25)
            Q3 = X.quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Remove outliers (keep rows where all values are within bounds)
            outlier_mask = ((X >= lower_bound) & (X <= upper_bound)).all(axis=1)
            X = X[outlier_mask]
            y = y[outlier_mask]
            subjects = subjects[outlier_mask]
            
            print(f"After outlier removal: {X.shape}")
        
        # Normalize features if requested
        if normalize:
            print("Normalizing features...")
            X_normalized = self.scaler.fit_transform(X)
            X = pd.DataFrame(X_normalized, columns=X.columns, index=X.index)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"Final dataset shape: {X.shape}")
        print(f"Number of classes: {len(np.unique(y_encoded))}")
        
        # Store processed data
        self.processed_data = {
            'features': X.values,
            'labels': y_encoded,
            'subjects': subjects.values,
            'feature_names': feature_cols,
            'class_names': self.label_encoder.classes_
        }
        
        return X.values, y_encoded, subjects.values
    
    def create_temporal_windows(self, window_size: int = 100, step_size: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create temporal windows for sequence learning"""
        if self.processed_data is None:
            raise ValueError("No processed data. Call preprocess_data() first.")
        
        print(f"Creating temporal windows (size={window_size}, step={step_size})...")
        
        X = self.processed_data['features']
        y = self.processed_data['labels']
        subjects = self.processed_data['subjects']
        
        windowed_X = []
        windowed_y = []
        windowed_subjects = []
        
        # Group by subject and activity for windowing
        unique_subjects = np.unique(subjects)
        
        for subject in unique_subjects:
            subject_mask = subjects == subject
            subject_X = X[subject_mask]
            subject_y = y[subject_mask]
            
            # Create windows for this subject
            for start in range(0, len(subject_X) - window_size + 1, step_size):
                window_X = subject_X[start:start + window_size]
                window_y = subject_y[start:start + window_size]
                
                # Use the most common activity in the window as the label
                unique_labels, counts = np.unique(window_y, return_counts=True)
                dominant_label = unique_labels[np.argmax(counts)]
                
                windowed_X.append(window_X)
                windowed_y.append(dominant_label)
                windowed_subjects.append(subject)
        
        windowed_X = np.array(windowed_X)
        windowed_y = np.array(windowed_y)
        windowed_subjects = np.array(windowed_subjects)
        
        print(f"Created {len(windowed_X)} windows")
        print(f"Window shape: {windowed_X.shape}")
        print(f"Classes in windowed data: {len(np.unique(windowed_y))}")
        
        return windowed_X, windowed_y, windowed_subjects
    
    def get_feature_groups_data(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract data grouped by feature types"""
        if self.processed_data is None:
            raise ValueError("No processed data available")
        
        feature_names = self.processed_data['feature_names']
        grouped_data = {}
        
        for group_name, indices in self.feature_groups.items():
            # Get columns for this feature group across all sensors
            group_indices = []
            
            for sensor in self.sensor_names:
                for idx in indices:
                    col_name = f"{self.feature_names[idx]}_{sensor}"
                    try:
                        col_idx = feature_names.index(col_name)
                        group_indices.append(col_idx)
                    except ValueError:
                        # Column not found, skip
                        continue
            
            if group_indices:
                if len(data.shape) == 3:  # Windowed data
                    group_data = data[:, :, group_indices]
                else:  # Regular data
                    group_data = data[:, group_indices]
                grouped_data[group_name] = group_data
        
        return grouped_data
    
    def analyze_correlations(self) -> Dict[str, np.ndarray]:
        """Analyze real feature correlations"""
        if self.processed_data is None:
            raise ValueError("No processed data. Call preprocess_data() first.")
        
        print("Analyzing feature correlations...")
        
        X = self.processed_data['features']
        feature_names = self.processed_data['feature_names']
        
        # Overall correlation matrix
        correlation_matrix = np.corrcoef(X.T)
        
        # Group-wise correlations
        group_correlations = {}
        grouped_data = self.get_feature_groups_data(X)
        
        for group_name, group_data in grouped_data.items():
            group_corr = np.corrcoef(group_data.T)
            group_correlations[group_name] = group_corr
            
            avg_corr = np.mean(np.abs(group_corr[np.triu_indices_from(group_corr, k=1)]))
            print(f"  {group_name}: average correlation = {avg_corr:.3f}")
        
        correlations = {
            'overall': correlation_matrix,
            'groups': group_correlations
        }
        
        return correlations


def test_data_loader():
    """Test the data loader with real SONAR data"""
    print("Testing SONAR Data Loader...")
    
    try:
        # Initialize loader
        loader = SONARDataLoader()
        
        # Load raw data
        raw_data = loader.load_raw_data(max_files=10)  # Start with 10 files for testing
        
        # Preprocess data
        X, y, subjects = loader.preprocess_data()
        
        # Create temporal windows
        X_windows, y_windows, subjects_windows = loader.create_temporal_windows()
        
        # Analyze correlations
        correlations = loader.analyze_correlations()
        
        # Get feature groups
        grouped_data = loader.get_feature_groups_data(X_windows)
        
        print("\n=== DATA LOADER TEST RESULTS ===")
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Processed data shape: {X.shape}")
        print(f"Windowed data shape: {X_windows.shape}")
        print(f"Number of subjects: {len(np.unique(subjects))}")
        print(f"Number of activities: {len(np.unique(y))}")
        print(f"Feature groups: {list(grouped_data.keys())}")
        
        for group_name, group_data in grouped_data.items():
            print(f"  {group_name}: {group_data.shape}")
        
        print("✓ Data loader test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_data_loader() 