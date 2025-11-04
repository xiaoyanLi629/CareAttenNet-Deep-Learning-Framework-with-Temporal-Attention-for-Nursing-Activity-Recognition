"""
Activity Signal Pattern Comparison Heatmap Generator

This script selects two different nursing activities from the same subject and generates 
a 140*time_length comparison heatmap, where the first 70 rows show the first activity's 
sensor signals and the last 70 rows show the second activity's sensor signals, 
demonstrating the differences in signal patterns between different activities.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class ActivityComparisonHeatmap:
    """Activity comparison heatmap generator"""
    
    def __init__(self, data_dir="../SONAR_ML"):
        self.data_dir = data_dir
        self.feature_columns = None
        self.selected_subject = None
        self.selected_activities = None
        
    def explore_dataset(self):
        """Explore dataset and select suitable subject and activities"""
        print("🔍 Exploring dataset structure...")
        
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))
        print(f"📁 Found {len(csv_files)} CSV files")
        
        subject_activities = {}
        subject_sample_counts = {}
        
        # Analyze activities for each subject
        for file_path in csv_files[:50]:  # Check first 50 files to save time
            try:
                filename = os.path.basename(file_path)
                if '_sub' not in filename:
                    continue
                    
                subject_id = filename.split('_sub')[1].split('.')[0]
                df = pd.read_csv(file_path)
                
                if 'activity' not in df.columns:
                    continue
                
                # Filter valid activities
                valid_activities = df[df['activity'] != 'null - activity']['activity'].value_counts()
                
                if subject_id not in subject_activities:
                    subject_activities[subject_id] = {}
                    subject_sample_counts[subject_id] = 0
                
                for activity, count in valid_activities.items():
                    if activity in subject_activities[subject_id]:
                        subject_activities[subject_id][activity] += count
                    else:
                        subject_activities[subject_id][activity] = count
                
                subject_sample_counts[subject_id] += len(df)
                
            except Exception as e:
                continue
        
        # Select best subject (most activity types and sufficient samples)
        best_subject = None
        best_score = 0
        
        print(f"\n👥 Subject activity analysis:")
        for subject_id, activities in subject_activities.items():
            # Calculate score: number of activities * minimum activity samples
            activity_counts = list(activities.values())
            if len(activity_counts) >= 2:
                score = len(activity_counts) * min(activity_counts)
                print(f"   sub{subject_id}: {len(activity_counts)} activities, min samples: {min(activity_counts)}, score: {score}")
                
                if score > best_score:
                    best_score = score
                    best_subject = subject_id
        
        if best_subject is None:
            raise ValueError("❌ No suitable subject data found")
        
        self.selected_subject = best_subject
        subject_acts = subject_activities[best_subject]
        
        # Select two activities with most samples and likely different patterns
        sorted_activities = sorted(subject_acts.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n✅ Selected subject: sub{best_subject}")
        print(f"📊 Activity distribution for this subject:")
        for activity, count in sorted_activities:
            print(f"   - {activity}: {count:,} samples")
        
        # Intelligently select two contrasting activities
        if len(sorted_activities) >= 2:
            # Prefer activity pairs that are likely to have different patterns
            activity_pairs = [
                ('change clothes', 'wash in bed'),
                ('kitchen preparation', 'documentation'), 
                ('serve food', 'comb hair'),
                ('clean up', 'wheelchair transfer'),
                ('push wheelchair', 'dental care')
            ]
            
            # Find available activity pairs
            available_activities = [act for act, _ in sorted_activities]
            selected_pair = None
            
            for act1, act2 in activity_pairs:
                if act1 in available_activities and act2 in available_activities:
                    selected_pair = (act1, act2)
                    break
            
            # If no preset combination, select top two activities
            if selected_pair is None:
                selected_pair = (sorted_activities[0][0], sorted_activities[1][0])
            
            self.selected_activities = selected_pair
            print(f"\n🎯 Selected activity comparison:")
            print(f"   Activity 1: {selected_pair[0]} ({subject_acts[selected_pair[0]]:,} samples)")
            print(f"   Activity 2: {selected_pair[1]} ({subject_acts[selected_pair[1]]:,} samples)")
        
        return self.selected_subject, self.selected_activities
    
    def load_activity_data(self, max_time_length=1000):
        """Load activity data for the selected subject"""
        print(f"\n📥 Loading activity data for subject sub{self.selected_subject}...")
        
        csv_files = glob.glob(os.path.join(self.data_dir, f"*_sub{self.selected_subject}.csv"))
        
        activity1_data = []
        activity2_data = []
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                
                if 'activity' not in df.columns:
                    continue
                
                # Extract data for both activities
                act1_mask = df['activity'] == self.selected_activities[0]
                act2_mask = df['activity'] == self.selected_activities[1]
                
                if act1_mask.any():
                    activity1_data.append(df[act1_mask])
                
                if act2_mask.any():
                    activity2_data.append(df[act2_mask])
                    
            except Exception as e:
                print(f"   ⚠️ Error loading file {file_path}: {e}")
                continue
        
        if not activity1_data or not activity2_data:
            raise ValueError("❌ Insufficient activity data found")
        
        # Combine data
        activity1_combined = pd.concat(activity1_data, ignore_index=True)
        activity2_combined = pd.concat(activity2_data, ignore_index=True)
        
        # Get feature columns (exclude metadata columns)
        exclude_cols = ['activity', 'SampleTimeFine', 'subject_id', 'file_id']
        self.feature_columns = [col for col in activity1_combined.columns if col not in exclude_cols]
        
        print(f"   Feature columns count: {len(self.feature_columns)}")
        print(f"   Activity 1 data: {len(activity1_combined)} samples")
        print(f"   Activity 2 data: {len(activity2_combined)} samples")
        
        # Ensure 70 features
        if len(self.feature_columns) != 70:
            self.feature_columns = self.feature_columns[:70]  # Take first 70 features
            print(f"   Adjusted to 70 feature columns")
        
        # Extract same length time segments
        min_length = min(len(activity1_combined), len(activity2_combined), max_time_length)
        
        activity1_features = activity1_combined[self.feature_columns].iloc[:min_length].values
        activity2_features = activity2_combined[self.feature_columns].iloc[:min_length].values
        
        # Handle missing values
        activity1_features = np.nan_to_num(activity1_features, nan=0.0)
        activity2_features = np.nan_to_num(activity2_features, nan=0.0)
        
        print(f"   ✅ Successfully extracted data: {min_length} time steps, 70 features")
        
        return activity1_features, activity2_features
    
    def create_comparison_heatmap(self, activity1_data, activity2_data, normalize=True):
        """Create 140*time_length comparison heatmap"""
        print(f"\n🎨 Generating activity comparison heatmap...")
        
        time_length = activity1_data.shape[0]
        print(f"   Time length: {time_length}")
        print(f"   Heatmap size: 140 × {time_length}")
        
        # Transpose data: from (time, features) to (features, time)
        activity1_transposed = activity1_data.T  # Shape: (70, time_length)
        activity2_transposed = activity2_data.T  # Shape: (70, time_length)
        
        # Combine into 140*time_length matrix
        combined_data = np.vstack([activity1_transposed, activity2_transposed])  # Shape: (140, time_length)
        
        print(f"   Combined data shape: {combined_data.shape}")
        
        # Standardization
        if normalize:
            scaler = StandardScaler()
            combined_data_scaled = scaler.fit_transform(combined_data)
        else:
            combined_data_scaled = combined_data
        
        # Create heatmap with improved visualization
        plt.figure(figsize=(24, 9))  # Half height for cells
        
        # Use green colormap without grid
        sns.heatmap(
            combined_data_scaled,
            cmap='Greens',
            center=0,
            cbar_kws={'label': 'Normalized Signal Intensity', 'shrink': 0.8, 'pad': 0.02},
            xticklabels=False,  # Too many time labels to show
            yticklabels=False   # Too many feature labels to show
        )
        
        # Add red separation line (half width)
        plt.axhline(y=70, color='red', linewidth=6, linestyle='-', alpha=1.0)
        
        # Add activity labels with English text (moved slightly right)
        plt.text(-25, 35, f'{self.selected_activities[0]}', 
                rotation=90, va='center', ha='right', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        plt.text(-25, 105, f'{self.selected_activities[1]}', 
                rotation=90, va='center', ha='right', fontsize=16, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        
        # Set title and labels in English
        plt.title(f'Nursing Activity Signal Pattern Comparison Heatmap\nSubject sub{self.selected_subject}: {self.selected_activities[0]} vs {self.selected_activities[1]}\nTime Length: {time_length} steps, Feature Dimension: 140 (70×2)', 
                  fontsize=18, fontweight='bold', pad=30)
        
        plt.xlabel(f'Time Steps (Total: {time_length} steps)', fontsize=14)
        plt.ylabel('Sensor Features (Rows 1-70: Activity 1, Rows 71-140: Activity 2)', fontsize=14)
        
        plt.tight_layout()
        
        # Save images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"activity_comparison_heatmap_v2_sub{self.selected_subject}_{self.selected_activities[0].replace(' ', '_')}_vs_{self.selected_activities[1].replace(' ', '_')}_{timestamp}.png"
        filepath = os.path.join("statistics/plots", filename)
        
        # Ensure directory exists
        os.makedirs("statistics/plots", exist_ok=True)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"   ✅ Heatmap saved: {filepath}")
        
        # Also save PDF version
        pdf_filepath = filepath.replace('.png', '.pdf')
        plt.savefig(pdf_filepath, bbox_inches='tight', facecolor='white')
        print(f"   📄 PDF version saved: {pdf_filepath}")
        
        # Close the figure to free memory
        plt.close()
        
        return filepath
    
    def generate_analysis_report(self, activity1_data, activity2_data):
        """Generate comparison analysis report"""
        print(f"\n📊 Generating analysis report...")
        
        # Calculate basic statistics
        act1_stats = {
            'mean': np.mean(activity1_data, axis=0),
            'std': np.std(activity1_data, axis=0),
            'max': np.max(activity1_data, axis=0),
            'min': np.min(activity1_data, axis=0)
        }
        
        act2_stats = {
            'mean': np.mean(activity2_data, axis=0),
            'std': np.std(activity2_data, axis=0),
            'max': np.max(activity2_data, axis=0),
            'min': np.min(activity2_data, axis=0)
        }
        
        # Calculate difference metrics
        mean_diff = np.abs(act1_stats['mean'] - act2_stats['mean'])
        std_diff = np.abs(act1_stats['std'] - act2_stats['std'])
        
        # Find features with largest differences
        top_diff_features = np.argsort(mean_diff)[-10:]  # Top 10 most different features
        
        report = {
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'subject_id': self.selected_subject,
            'activities': self.selected_activities,
            'data_shape': activity1_data.shape,
            'mean_difference_top10': [
                {
                    'feature_index': int(idx),
                    'feature_name': self.feature_columns[idx] if idx < len(self.feature_columns) else f'feature_{idx}',
                    'activity1_mean': float(act1_stats['mean'][idx]),
                    'activity2_mean': float(act2_stats['mean'][idx]),
                    'absolute_difference': float(mean_diff[idx])
                }
                for idx in reversed(top_diff_features)
            ],
            'overall_statistics': {
                'activity1': {
                    'mean_signal_strength': float(np.mean(act1_stats['mean'])),
                    'signal_variability': float(np.mean(act1_stats['std'])),
                    'signal_range': float(np.mean(act1_stats['max'] - act1_stats['min']))
                },
                'activity2': {
                    'mean_signal_strength': float(np.mean(act2_stats['mean'])),
                    'signal_variability': float(np.mean(act2_stats['std'])),
                    'signal_range': float(np.mean(act2_stats['max'] - act2_stats['min']))
                }
            }
        }
        
        # Save report
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"activity_comparison_report_v2_sub{self.selected_subject}_{timestamp}.json"
        report_filepath = os.path.join("statistics/reports", report_filename)
        
        os.makedirs("statistics/reports", exist_ok=True)
        
        with open(report_filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Analysis report saved: {report_filepath}")
        
        # Print key findings
        print(f"\n🔍 Key Findings:")
        print(f"   Comparison activities: {self.selected_activities[0]} vs {self.selected_activities[1]}")
        print(f"   Data dimensions: {activity1_data.shape}")
        print(f"   Top 3 most different features:")
        for i, feature_info in enumerate(report['mean_difference_top10'][:3]):
            print(f"     {i+1}. {feature_info['feature_name']}: difference {feature_info['absolute_difference']:.4f}")
        
        return report_filepath

def main():
    """Main function"""
    print("="*80)
    print("Nursing Activity Signal Pattern Comparison Heatmap Generator")
    print("="*80)
    
    try:
        # Create analyzer
        analyzer = ActivityComparisonHeatmap()
        
        # 1. Explore dataset and select subject and activities
        subject, activities = analyzer.explore_dataset()
        
        # 2. Load activity data with longer time length
        activity1_data, activity2_data = analyzer.load_activity_data(max_time_length=1000)
        
        # 3. Generate comparison heatmap
        heatmap_path = analyzer.create_comparison_heatmap(activity1_data, activity2_data, normalize=True)
        
        # 4. Generate analysis report
        report_path = analyzer.generate_analysis_report(activity1_data, activity2_data)
        
        print(f"\n🎉 Analysis completed!")
        print(f"📊 Heatmap file: {heatmap_path}")
        print(f"📋 Analysis report: {report_path}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 