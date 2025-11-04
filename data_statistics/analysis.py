#!/usr/bin/env python3
"""
SONaR Dataset Enhanced Statistical Analysis Tool
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import json
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import squarify
warnings.filterwarnings('ignore')

# Set matplotlib to use English locale and proper font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.autolayout'] = True

def extract_subject_id(filename):
    """Extract subject ID from filename (e.g., '123_sub7.csv' -> 'sub7')"""
    try:
        if '_sub' in filename:
            return filename.split('_sub')[1].split('.')[0]
        return 'unknown'
    except:
        return 'unknown'

def calculate_statistical_measures(data, feature_cols):
    """Calculate comprehensive statistical measures for features"""
    stats_dict = {}
    
    for col in feature_cols:
        if col in data.columns and data[col].dtype in [np.float64, np.int64]:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                stats_dict[col] = {
                    'count': len(col_data),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'median': float(col_data.median()),
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75)),
                    'skewness': float(stats.skew(col_data)),
                    'kurtosis': float(stats.kurtosis(col_data)),
                    'range': float(col_data.max() - col_data.min()),
                    'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25))
                }
    
    return stats_dict

def generate_dataset_treemaps(data, activity_counts, subject_stats, activity_stats, feature_cols, output_dir="statistics"):
    """
    Generate comprehensive treemap visualizations for dataset analysis
    
    Args:
        data: Combined dataset
        activity_counts: Activity count statistics
        subject_stats: Subject-level statistics
        activity_stats: Activity-level statistics  
        feature_cols: List of feature column names
        output_dir: Output directory for saving plots
    """
    print("\n🌳 Generating Comprehensive Treemap Visualizations...")
    
    # Create treemap output directory
    treemap_dir = f"{output_dir}/treemaps"
    os.makedirs(treemap_dir, exist_ok=True)
    
    # 1. Activity Distribution Treemap (Static - Matplotlib + Squarify)
    print("  📊 Creating Activity Distribution Treemap...")
    
    # Prepare data for activity treemap - exclude specific activities
    excluded_activities = ['wash hair', 'put accessories', 'blow-dry', 'put medication', 'comb hair', 'pour drinks', 'collect dishes']
    filtered_activity_counts = activity_counts[~activity_counts.index.isin(excluded_activities)]
    
    activity_names = filtered_activity_counts.index.tolist()
    activity_values = filtered_activity_counts.values.tolist()
    
    # Create color map based on activity frequency
    max_val = max(activity_values)
    colors = [plt.cm.viridis(val/max_val) for val in activity_values]
    
    # Create matplotlib treemap
    fig, ax = plt.subplots(figsize=(16, 12))
    squarify.plot(sizes=activity_values, 
                  label=[f"{name}\n({val:,} samples)" for name, val in zip(activity_names, activity_values)],
                  color=colors,
                  alpha=0.8,
                  text_kwargs={'fontsize': 16, 'weight': 'bold'})
    
    plt.title('Nursing Activity Distribution Treemap\n(Size proportional to sample count)', 
              fontsize=20, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{treemap_dir}/activity_distribution_treemap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Interactive Activity Distribution Treemap (Plotly)
    print("  🌐 Creating Interactive Activity Distribution Treemap...")
    
    # Create interactive plotly treemap using filtered data
    fig_interactive = go.Figure(go.Treemap(
        values=activity_values,
        parents=[""] * len(activity_names),
        labels=activity_names,
        text=[f"{name}<br>{val:,} samples" for name, val in zip(activity_names, activity_values)],
        texttemplate="<b>%{label}</b><br>%{value:,} samples",
        hovertemplate='<b>%{label}</b><br>Samples: %{value:,}<br>Percentage: %{percentParent}<extra></extra>',
        marker=dict(
            colorscale='Viridis',
            colorbar=dict(title="Sample Count")
        )
    ))
    
    fig_interactive.update_layout(
        title={'text': "Interactive Nursing Activity Distribution Treemap", 'x': 0.5},
        font_size=12,
        width=1200,
        height=800
    )
    
    fig_interactive.write_html(f'{treemap_dir}/interactive_activity_treemap.html')
    
    # 3. Subject-Activity Distribution Hierarchical Treemap
    print("  👥 Creating Subject-Activity Hierarchical Treemap...")
    
    # Prepare hierarchical data for subject-activity treemap
    hierarchical_data = []
    
    for subject_id, stats in subject_stats.items():
        subject_total = stats['total_samples']
        activity_dist = stats['activity_distribution']
        
        # Add subject as parent
        hierarchical_data.append({
            'ids': subject_id,
            'labels': f"Subject {subject_id}",
            'parents': "",
            'values': subject_total
        })
        
        # Add activities for this subject
        for activity, count in activity_dist.items():
            if count > 100:  # Only show activities with sufficient samples
                hierarchical_data.append({
                    'ids': f"{subject_id}_{activity}",
                    'labels': f"{activity} ({count:,})",
                    'parents': subject_id,
                    'values': count
                })
    
    # Create hierarchical treemap
    if hierarchical_data:
        df_hierarchical = pd.DataFrame(hierarchical_data)
        
        fig_hierarchical = go.Figure(go.Treemap(
            ids=df_hierarchical['ids'],
            labels=df_hierarchical['labels'],
            parents=df_hierarchical['parents'],
            values=df_hierarchical['values'],
            branchvalues="total",
            maxdepth=2,
            texttemplate="<b>%{label}</b><br>%{value:,} samples",
            textfont_size=9
        ))
        
        fig_hierarchical.update_layout(
            title={'text': "Subject-Activity Hierarchical Distribution", 'x': 0.5},
            font_size=12,
            width=1200,
            height=800
        )
        
        fig_hierarchical.write_html(f'{treemap_dir}/subject_activity_hierarchical_treemap.html')
    
    # 4. Feature Group Treemap (by sensor type)
    print("  🔧 Creating Feature Group Treemap...")
    
    # Categorize features by sensor type
    feature_groups = {
        'Quaternion Features': [],
        'Angular Velocity Features': [],
        'Linear Velocity Features': [],
        'Magnetic Field Features': [],
        'Other Features': []
    }
    
    for col in feature_cols:
        if 'activity' not in col.lower() and 'time' not in col.lower() and 'subject' not in col.lower():
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['quat', 'quaternion', 'q_']):
                feature_groups['Quaternion Features'].append(col)
            elif any(keyword in col_lower for keyword in ['gyro', 'angular', 'rotation']):
                feature_groups['Angular Velocity Features'].append(col)
            elif any(keyword in col_lower for keyword in ['accel', 'linear', 'velocity', 'vel']):
                feature_groups['Linear Velocity Features'].append(col)
            elif any(keyword in col_lower for keyword in ['mag', 'magnetic', 'compass']):
                feature_groups['Magnetic Field Features'].append(col)
            else:
                feature_groups['Other Features'].append(col)
    
    # Calculate feature group statistics
    feature_group_stats = {}
    numeric_data = data.select_dtypes(include=[np.number])
    
    for group_name, features in feature_groups.items():
        if features:
            group_features = [f for f in features if f in numeric_data.columns]
            if group_features:
                group_data = numeric_data[group_features]
                feature_group_stats[group_name] = {
                    'count': len(group_features),
                    'avg_variance': group_data.var().mean(),
                    'total_variance': group_data.var().sum(),
                    'features': group_features
                }
    
    # Create feature group treemap based on total variance
    if feature_group_stats:
        group_names = list(feature_group_stats.keys())
        group_variances = [stats['total_variance'] for stats in feature_group_stats.values()]
        group_counts = [stats['count'] for stats in feature_group_stats.values()]
        
        # Create static treemap for feature groups
        fig, ax = plt.subplots(figsize=(14, 10))
        colors_groups = plt.cm.Set3(np.linspace(0, 1, len(group_names)))
        
        squarify.plot(sizes=group_variances,
                      label=[f"{name}\n{count} features\nVar: {var:.2f}" 
                            for name, count, var in zip(group_names, group_counts, group_variances)],
                      color=colors_groups,
                      alpha=0.8,
                      text_kwargs={'fontsize': 14, 'weight': 'bold'})
        
        plt.title('Sensor Feature Groups by Total Variance\n(Size proportional to feature variance)', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{treemap_dir}/feature_groups_treemap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interactive feature group treemap
        feature_labels = [f"{name} ({count} features)" for name, count in zip(group_names, group_counts)]
        fig_features = go.Figure(go.Treemap(
            values=group_variances,
            parents=[""] * len(group_names),
            labels=feature_labels,
            texttemplate="<b>%{label}</b><br>Variance: %{value:.4f}",
            hovertemplate='<b>%{label}</b><br>Variance: %{value:.4f}<extra></extra>',
            marker=dict(
                colorscale='Plasma',
                colorbar=dict(title="Total Variance")
            )
        ))
        
        fig_features.update_layout(
            title={'text': "Interactive Feature Groups by Variance", 'x': 0.5},
            font_size=12,
            width=1000,
            height=700
        )
        
        fig_features.write_html(f'{treemap_dir}/interactive_feature_groups_treemap.html')
    
    # 5. Combined Dataset Overview Treemap
    print("  📋 Creating Dataset Overview Treemap...")
    
    # Create comprehensive overview treemap
    overview_data = [
        {'name': 'Data Samples', 'value': len(data), 'parent': 'Dataset'},
        {'name': 'Dataset', 'value': 0, 'parent': ''},
        {'name': 'Features', 'value': len(feature_cols), 'parent': 'Dataset'},
        {'name': 'Subjects', 'value': len(subject_stats), 'parent': 'Dataset'},
        {'name': 'Activities', 'value': len(filtered_activity_counts), 'parent': 'Dataset'}
    ]
    
    # Add activity breakdown using filtered activities
    for activity, count in filtered_activity_counts.head(10).items():
        overview_data.append({
            'name': f"{activity[:20]}..." if len(activity) > 20 else activity,
            'value': count,
            'parent': 'Activities'
        })
    
    # Create overview treemap
    df_overview = pd.DataFrame(overview_data)
    
    fig_overview = go.Figure(go.Treemap(
        ids=df_overview['name'],
        labels=df_overview['name'],
        parents=df_overview['parent'],
        values=df_overview['value'],
        branchvalues="total",
        maxdepth=3,
        texttemplate="<b>%{label}</b><br>%{value:,}",
        textfont_size=10
    ))
    
    fig_overview.update_layout(
        title={'text': "SONaR Dataset Overview Treemap", 'x': 0.5},
        font_size=12,
        width=1200,
        height=800
    )
    
    fig_overview.write_html(f'{treemap_dir}/dataset_overview_treemap.html')
    
    print("  ✅ Treemap generation completed!")
    print(f"     📁 Static treemaps saved to: {treemap_dir}/")
    print(f"     🌐 Interactive treemaps saved to: {treemap_dir}/")
    
    return {
        'treemap_files': {
            'activity_distribution': f'{treemap_dir}/activity_distribution_treemap.png',
            'interactive_activity': f'{treemap_dir}/interactive_activity_treemap.html',
            'subject_activity_hierarchical': f'{treemap_dir}/subject_activity_hierarchical_treemap.html',
            'feature_groups': f'{treemap_dir}/feature_groups_treemap.png',
            'interactive_feature_groups': f'{treemap_dir}/interactive_feature_groups_treemap.html',
            'dataset_overview': f'{treemap_dir}/dataset_overview_treemap.html'
        },
        'feature_group_analysis': feature_group_stats if 'feature_group_stats' in locals() else {},
        'treemap_statistics': {
            'total_activities_visualized': len(activity_names),
            'total_subjects_analyzed': len(subject_stats),
            'feature_groups_identified': len(feature_groups) if feature_groups else 0,
            'excluded_activities': excluded_activities
        }
    }

def generate_feature_correlation_heatmap(data, feature_cols, output_dir="statistics", sample_size=10000):
    """
    Generate comprehensive correlation matrix heatmap for all 70 sensor features
    
    Args:
        data: Combined dataset
        feature_cols: List of feature column names
        output_dir: Output directory for saving plots
        sample_size: Sample size for correlation computation (to handle large datasets)
    """
    print("\n🔥 Generating 70-Feature Correlation Matrix Heatmap...")
    
    # Filter to only numeric features (exclude categorical columns)
    exclude_cols = ['activity', 'SampleTimeFine', 'subject_id', 'file_id']
    numeric_feature_cols = [col for col in feature_cols if col not in exclude_cols]
    
    # Sample data if dataset is too large
    if len(data) > sample_size:
        print(f"  📊 Sampling {sample_size:,} rows from {len(data):,} total samples for correlation analysis")
        sampled_data = data.sample(n=sample_size, random_state=42)
    else:
        sampled_data = data
        print(f"  📊 Using all {len(data):,} samples for correlation analysis")
    
    # Select numeric features and handle missing values
    features_data = sampled_data[numeric_feature_cols].select_dtypes(include=[np.number])
    features_data = features_data.fillna(features_data.mean())
    
    print(f"  🎯 Computing correlation matrix for {len(features_data.columns)} features...")
    
    # Compute correlation matrix
    correlation_matrix = features_data.corr()
    
    # Replace any remaining NaN values with 0
    correlation_matrix = correlation_matrix.fillna(0)
    
    print(f"  ✅ Correlation matrix computed: {correlation_matrix.shape}")
    
    # Create the heatmap figure
    plt.figure(figsize=(20, 18))
    
    # Create mask for upper triangle (optional - shows full matrix)
    # mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    # Generate the heatmap with enhanced styling
    heatmap = sns.heatmap(
        correlation_matrix,
        # mask=mask,
        annot=False,  # Don't show values due to many features
        cmap='RdBu_r',  # Red-Blue colormap (red=positive, blue=negative)
        center=0,  # Center colormap at 0
        square=True,  # Make cells square
        linewidths=0.1,  # Thin lines between cells
        cbar_kws={
            "shrink": 0.8,
            "orientation": "vertical",
            "label": "Correlation Coefficient"
        },
        xticklabels=True,
        yticklabels=True
    )
    
    # Enhance the plot appearance
    plt.title('Sensor Feature Correlation Matrix Heatmap\n70 Features from Nursing Activity Dataset', 
              fontsize=18, fontweight='bold', pad=20)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save high-resolution heatmap
    heatmap_path = f'{output_dir}/plots/feature_correlation_heatmap_70features.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  💾 Saved correlation heatmap: {heatmap_path}")
    
    # Also save as PDF for publications
    pdf_path = f'{output_dir}/plots/feature_correlation_heatmap_70features.pdf'
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"  📄 Saved PDF version: {pdf_path}")
    
    plt.close()
    
    # Generate a simplified clustered heatmap for better visualization
    print("  🎨 Generating clustered correlation heatmap...")
    
    plt.figure(figsize=(18, 16))
    
    # Create clustered heatmap
    clustered_heatmap = sns.clustermap(
        correlation_matrix,
        method='average',  # Linkage method for clustering
        metric='correlation',  # Distance metric
        cmap='RdBu_r',
        center=0,
        annot=False,
        square=True,
        linewidths=0.1,
        figsize=(18, 16),
        xticklabels=False,  # Remove x-axis labels
        yticklabels=False,  # Remove y-axis labels
        cbar=False  # Remove colorbar
    )
    
    # Remove any remaining colorbar artifacts
    for ax in clustered_heatmap.fig.axes:
        if hasattr(ax, 'get_position'):
            pos = ax.get_position()
            # Hide any axes that might be colorbar remnants (typically positioned on the right)
            if pos.x0 > 0.9:  # Axes positioned far right are likely colorbar remnants
                ax.set_visible(False)
    
    # Set title
    clustered_heatmap.fig.suptitle(
        'Clustered Sensor Feature Correlation Matrix\n70 Features Grouped by Similarity', 
        fontsize=16, fontweight='bold', y=0.98
    )
    
    # Save clustered version
    clustered_path = f'{output_dir}/plots/feature_correlation_heatmap_clustered.png'
    clustered_heatmap.savefig(clustered_path, dpi=300, bbox_inches='tight')
    print(f"  💾 Saved clustered heatmap: {clustered_path}")
    
    plt.close()
    
    # Generate summary statistics about correlations
    print("  📈 Computing correlation statistics...")
    
    # Get correlation values (excluding diagonal)
    corr_values = correlation_matrix.values
    np.fill_diagonal(corr_values, np.nan)  # Remove self-correlations
    corr_values_flat = corr_values.flatten()
    corr_values_clean = corr_values_flat[~np.isnan(corr_values_flat)]
    
    # Find strongest correlations
    strong_correlations = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:  # Threshold for strong correlation
                strong_correlations.append({
                    'feature_1': correlation_matrix.columns[i],
                    'feature_2': correlation_matrix.columns[j],
                    'correlation': float(corr_val),
                    'abs_correlation': float(abs(corr_val))
                })
    
    # Sort by absolute correlation
    strong_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
    
    # Correlation summary statistics
    correlation_stats = {
        'total_feature_pairs': len(corr_values_clean),
        'strong_correlations_count': len(strong_correlations),
        'max_correlation': float(np.max(np.abs(corr_values_clean))),
        'mean_abs_correlation': float(np.mean(np.abs(corr_values_clean))),
        'std_correlation': float(np.std(corr_values_clean)),
        'correlations_above_05': len([c for c in corr_values_clean if abs(c) > 0.5]),
        'correlations_above_07': len([c for c in corr_values_clean if abs(c) > 0.7]),
        'correlations_above_09': len([c for c in corr_values_clean if abs(c) > 0.9]),
        'top_20_correlations': strong_correlations[:20]
    }
    
    # Save correlation analysis results
    correlation_results_path = f'{output_dir}/reports/feature_correlation_analysis.json'
    with open(correlation_results_path, 'w', encoding='utf-8') as f:
        json.dump(correlation_stats, f, indent=2, ensure_ascii=False)
    print(f"  📋 Saved correlation analysis: {correlation_results_path}")
    
    # Save correlation matrix as CSV for further analysis
    correlation_csv_path = f'{output_dir}/detailed_stats/feature_correlation_matrix.csv'
    correlation_matrix.to_csv(correlation_csv_path)
    print(f"  📊 Saved correlation matrix CSV: {correlation_csv_path}")
    
    print(f"  ✅ Feature correlation analysis completed!")
    print(f"     🔗 Found {len(strong_correlations)} strong correlations (|r| > 0.5)")
    print(f"     📈 Max correlation: {correlation_stats['max_correlation']:.3f}")
    print(f"     📊 Mean |correlation|: {correlation_stats['mean_abs_correlation']:.3f}")
    
    return correlation_matrix, correlation_stats

def analyze_sonar_dataset_enhanced(data_dir="../SONAR_ML", max_files=253, output_dir="statistics"):
    """Enhanced analysis of SONaR dataset with detailed statistics"""
    
    print("🔬 SONaR Dataset Enhanced Statistical Analysis")
    print("=" * 60)
    
    # 1. Create output directories
    os.makedirs(f"{output_dir}/plots", exist_ok=True)
    os.makedirs(f"{output_dir}/reports", exist_ok=True)
    os.makedirs(f"{output_dir}/detailed_stats", exist_ok=True)
    
    # 2. Load dataset
    print("📂 Loading dataset...")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Found {len(csv_files)} CSV files")
    
    all_data = []
    file_info = []
    subject_data = {}
    
    for i, file_path in enumerate(csv_files[:max_files]):
        try:
            df = pd.read_csv(file_path)
            filename = os.path.basename(file_path)
            subject_id = extract_subject_id(filename)
            
            info = {
                'filename': filename,
                'subject_id': subject_id,
                'rows': len(df),
                'columns': len(df.columns),
                'has_activity': 'activity' in df.columns,
                'has_time': 'SampleTimeFine' in df.columns
            }
            
            if 'activity' in df.columns:
                # Filter valid activities
                valid_data = df[df['activity'] != 'null - activity'].dropna(subset=['activity'])
                if len(valid_data) > 100:
                    # Add subject ID to data
                    valid_data = valid_data.copy()
                    valid_data['subject_id'] = subject_id
                    all_data.append(valid_data)
                    
                    # Store per-subject data
                    if subject_id not in subject_data:
                        subject_data[subject_id] = []
                    subject_data[subject_id].append(valid_data)
                    
                    info['activities'] = valid_data['activity'].nunique()
                    info['valid_samples'] = len(valid_data)
            
            file_info.append(info)
            
            if i % 20 == 0:
                print(f"  Processing: {i+1}/{min(len(csv_files), max_files)}")
                
        except Exception as e:
            print(f"  Error processing {file_path}: {e}")
            continue
    
    # 3. Combine data
    if not all_data:
        print("❌ No valid data found")
        return None
    
    combined_data = pd.concat(all_data, ignore_index=True)
    print(f"✓ Combined {len(combined_data)} samples from {len(subject_data)} subjects")
    
    # 4. Basic analysis
    print("\n📊 Conducting enhanced statistical analysis...")
    
    # Activity and feature analysis
    activity_counts = combined_data['activity'].value_counts()
    feature_cols = [col for col in combined_data.columns 
                   if col not in ['activity', 'SampleTimeFine', 'subject_id']]
    features_data = combined_data[feature_cols]
    
    # 5. Subject-specific analysis
    print("👥 Analyzing per-subject statistics...")
    subject_stats = {}
    
    for subject_id in subject_data:
        subject_combined = pd.concat(subject_data[subject_id], ignore_index=True)
        subject_activities = subject_combined['activity'].value_counts()
        
        subject_stats[subject_id] = {
            'total_samples': len(subject_combined),
            'unique_activities': len(subject_activities),
            'activity_distribution': subject_activities.to_dict(),
            'files_count': len(subject_data[subject_id]),
            'feature_statistics': calculate_statistical_measures(subject_combined, feature_cols)
        }
    
    # 6. Activity-specific analysis
    print("🎯 Analyzing per-activity statistics...")
    activity_stats = {}
    
    for activity in activity_counts.index:
        activity_data = combined_data[combined_data['activity'] == activity]
        subjects_for_activity = activity_data['subject_id'].nunique()
        
        activity_stats[activity] = {
            'total_samples': len(activity_data),
            'subjects_count': subjects_for_activity,
            'subjects_list': activity_data['subject_id'].unique().tolist(),
            'avg_samples_per_subject': len(activity_data) / subjects_for_activity if subjects_for_activity > 0 else 0,
            'feature_statistics': calculate_statistical_measures(activity_data, feature_cols)
        }
    
    # 7. Overall feature correlation analysis
    print("🔗 Analyzing feature correlations...")
    numeric_features = features_data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_features.corr()
    
    # Find highly correlated features
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_val = correlation_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append({
                    'feature1': correlation_matrix.columns[i],
                    'feature2': correlation_matrix.columns[j],
                    'correlation': float(corr_val)
                })

    # 7.1. Generate comprehensive feature correlation heatmap
    print("🔥 Generating comprehensive feature correlation heatmaps...")
    feature_correlation_matrix, correlation_stats = generate_feature_correlation_heatmap(
        combined_data, feature_cols, output_dir, sample_size=50000
    )
    
    # 8. Generate comprehensive results
    analysis_results = {
        'dataset_info': {
            'total_files': len(csv_files),
            'processed_files': len([f for f in file_info if f.get('valid_samples', 0) > 0]),
            'total_samples': len(combined_data),
            'total_features': len(feature_cols),
            'total_subjects': len(subject_data),
            'analysis_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        'activity_analysis': {
            'total_activities': len(activity_counts),
            'activity_distribution': activity_counts.to_dict(),
            'most_frequent_activity': activity_counts.index[0],
            'least_frequent_activity': activity_counts.index[-1],
            'activities_with_1000_samples': len(activity_counts[activity_counts >= 1000]),
            'activities_with_500_samples': len(activity_counts[activity_counts >= 500]),
            'detailed_activity_stats': activity_stats
        },
        'subject_analysis': {
            'total_subjects': len(subject_data),
            'avg_samples_per_subject': np.mean([subject_stats[s]['total_samples'] for s in subject_stats]),
            'std_samples_per_subject': np.std([subject_stats[s]['total_samples'] for s in subject_stats]),
            'avg_activities_per_subject': np.mean([subject_stats[s]['unique_activities'] for s in subject_stats]),
            'std_activities_per_subject': np.std([subject_stats[s]['unique_activities'] for s in subject_stats]),
            'detailed_subject_stats': subject_stats
        },
        'feature_analysis': {
            'feature_count': len(feature_cols),
            'missing_values': features_data.isnull().sum().sum(),
            'missing_rate': features_data.isnull().sum().sum() / (len(features_data) * len(feature_cols)),
            'numeric_features': len(numeric_features.columns),
            'overall_feature_stats': calculate_statistical_measures(combined_data, feature_cols),
            'high_correlation_pairs': high_corr_pairs,
            'correlation_matrix_stats': correlation_stats
        },
        'data_quality': {
            'duplicate_rows': combined_data.duplicated().sum(),
            'completeness': 1 - (features_data.isnull().sum().sum() / (len(features_data) * len(feature_cols)))
        }
    }
    
    # 9. Generate enhanced visualizations
    print("\n📈 Generating enhanced visualization charts...")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 16))
    
    # Activity distribution
    plt.subplot(3, 3, 1)
    top_15 = activity_counts.head(15)
    plt.bar(range(len(top_15)), top_15.values, color='skyblue')
    plt.title('Top 15 Activity Distribution', fontweight='bold', fontsize=12)
    plt.xlabel('Activities')
    plt.ylabel('Sample Count')
    plt.xticks(range(len(top_15)), top_15.index, rotation=45, ha='right')
    
    # Subject distribution
    plt.subplot(3, 3, 2)
    subject_sample_counts = [subject_stats[s]['total_samples'] for s in subject_stats]
    plt.hist(subject_sample_counts, bins=15, color='lightgreen', edgecolor='black')
    plt.title('Samples Distribution per Subject', fontweight='bold', fontsize=12)
    plt.xlabel('Number of Samples')
    plt.ylabel('Number of Subjects')
    
    # Activities per subject
    plt.subplot(3, 3, 3)
    subject_activity_counts = [subject_stats[s]['unique_activities'] for s in subject_stats]
    plt.hist(subject_activity_counts, bins=10, color='orange', edgecolor='black')
    plt.title('Activities per Subject Distribution', fontweight='bold', fontsize=12)
    plt.xlabel('Number of Activities')
    plt.ylabel('Number of Subjects')
    
    # Correlation heatmap (top 10 features)
    plt.subplot(3, 3, 4)
    if len(numeric_features.columns) > 10:
        top_features = numeric_features.columns[:10]
        corr_subset = correlation_matrix.loc[top_features, top_features]
    else:
        corr_subset = correlation_matrix
    
    sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0, 
                square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
    plt.title('Feature Correlation Matrix', fontweight='bold', fontsize=12)
    
    # Activity sample statistics
    plt.subplot(3, 3, 5)
    activity_sample_counts = [activity_stats[act]['total_samples'] for act in activity_stats]
    plt.boxplot(activity_sample_counts)
    plt.title('Activity Sample Count Distribution', fontweight='bold', fontsize=12)
    plt.ylabel('Sample Count')
    plt.xlabel('Activities (Boxplot)')
    
    # Subject participation in activities
    plt.subplot(3, 3, 6)
    activity_subject_counts = [activity_stats[act]['subjects_count'] for act in activity_stats]
    plt.bar(range(len(activity_subject_counts)), sorted(activity_subject_counts, reverse=True), 
            color='mediumpurple')
    plt.title('Subject Participation by Activity', fontweight='bold', fontsize=12)
    plt.xlabel('Activities (Sorted)')
    plt.ylabel('Number of Subjects')
    
    # Feature variance analysis
    plt.subplot(3, 3, 7)
    if len(feature_cols) > 0:
        feature_vars = [analysis_results['feature_analysis']['overall_feature_stats'].get(f, {}).get('std', 0) 
                       for f in feature_cols[:20]]  # Top 20 features
        plt.bar(range(len(feature_vars)), feature_vars, color='salmon')
        plt.title('Feature Standard Deviations', fontweight='bold', fontsize=12)
        plt.xlabel('Features')
        plt.ylabel('Standard Deviation')
    
    # Data quality overview
    plt.subplot(3, 3, 8)
    quality_metrics = ['Completeness', 'Non-duplicate Rate']
    quality_values = [
        analysis_results['data_quality']['completeness'],
        1 - (analysis_results['data_quality']['duplicate_rows'] / len(combined_data))
    ]
    plt.bar(quality_metrics, quality_values, color=['lightblue', 'lightcoral'])
    plt.title('Data Quality Metrics', fontweight='bold', fontsize=12)
    plt.ylabel('Rate')
    plt.ylim(0, 1)
    
    # Summary statistics
    plt.subplot(3, 3, 9)
    summary_labels = ['Subjects', 'Activities', 'Features', 'Files']
    summary_values = [
        len(subject_data),
        len(activity_counts),
        len(feature_cols),
        analysis_results['dataset_info']['processed_files']
    ]
    plt.bar(summary_labels, summary_values, color='gold')
    plt.title('Dataset Summary', fontweight='bold', fontsize=12)
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/plots/enhanced_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 10. Generate comprehensive treemaps
    print("\n🌳 Generating comprehensive treemap visualizations...")
    treemap_results = generate_dataset_treemaps(
        data=combined_data,
        activity_counts=activity_counts,
        subject_stats=subject_stats,
        activity_stats=activity_stats,
        feature_cols=feature_cols,
        output_dir=output_dir
    )
    
    # Add treemap results to analysis results
    analysis_results['treemap_analysis'] = treemap_results
    
    # 11. Save detailed results
    print("\n💾 Saving detailed analysis results...")
    
    # Save comprehensive results
    with open(f'{output_dir}/reports/enhanced_analysis_results.json', 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
    
    # Save subject statistics separately
    with open(f'{output_dir}/detailed_stats/subject_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(subject_stats, f, indent=2, ensure_ascii=False, default=str)
    
    # Save activity statistics separately
    with open(f'{output_dir}/detailed_stats/activity_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(activity_stats, f, indent=2, ensure_ascii=False, default=str)
    
    # Generate comprehensive text report
    report_lines = [
        "SONaR Nursing Activity Dataset - Enhanced Analysis Report",
        "=" * 60,
        f"Analysis Time: {analysis_results['dataset_info']['analysis_timestamp']}",
        "",
        "DATASET OVERVIEW:",
        f"  Total Files: {analysis_results['dataset_info']['total_files']}",
        f"  Processed Files: {analysis_results['dataset_info']['processed_files']}",
        f"  Total Samples: {analysis_results['dataset_info']['total_samples']:,}",
        f"  Total Features: {analysis_results['dataset_info']['total_features']}",
        f"  Total Subjects: {analysis_results['dataset_info']['total_subjects']}",
        "",
        "SUBJECT ANALYSIS:",
        f"  Number of Subjects: {analysis_results['subject_analysis']['total_subjects']}",
        f"  Avg Samples per Subject: {analysis_results['subject_analysis']['avg_samples_per_subject']:.1f} ± {analysis_results['subject_analysis']['std_samples_per_subject']:.1f}",
        f"  Avg Activities per Subject: {analysis_results['subject_analysis']['avg_activities_per_subject']:.1f} ± {analysis_results['subject_analysis']['std_activities_per_subject']:.1f}",
        "",
        "ACTIVITY ANALYSIS:",
        f"  Total Activity Types: {analysis_results['activity_analysis']['total_activities']}",
        f"  Most Frequent Activity: {analysis_results['activity_analysis']['most_frequent_activity']}",
        f"  Activities with ≥1000 samples: {analysis_results['activity_analysis']['activities_with_1000_samples']}",
        f"  Activities with ≥500 samples: {analysis_results['activity_analysis']['activities_with_500_samples']}",
        "",
        "FEATURE ANALYSIS:",
        f"  Total Features: {analysis_results['feature_analysis']['feature_count']}",
        f"  Numeric Features: {analysis_results['feature_analysis']['numeric_features']}",
        f"  High Correlations Found: {len(analysis_results['feature_analysis']['high_correlation_pairs'])}",
        f"  Strong Correlations (|r|>0.5): {correlation_stats['strong_correlations_count']}",
        f"  Max Feature Correlation: {correlation_stats['max_correlation']:.3f}",
        "",
        "DATA QUALITY:",
        f"  Data Completeness: {analysis_results['data_quality']['completeness']:.2%}",
        f"  Duplicate Rows: {analysis_results['data_quality']['duplicate_rows']}",
        f"  Missing Value Rate: {analysis_results['feature_analysis']['missing_rate']:.2%}",
        "",
        "KEY INSIGHTS:",
        f"  1. Dataset contains {analysis_results['activity_analysis']['total_activities']} nursing activities from {analysis_results['subject_analysis']['total_subjects']} subjects",
        f"  2. {analysis_results['activity_analysis']['activities_with_1000_samples']} activities have sufficient samples (≥1000) for reliable training",
        f"  3. Data quality is high with {analysis_results['data_quality']['completeness']:.1%} completeness",
        f"  4. {len(analysis_results['feature_analysis']['high_correlation_pairs'])} feature pairs show high correlation (>0.8)",
        f"  5. {correlation_stats['strong_correlations_count']} sensor feature pairs show strong correlation (>0.5)",
        f"  6. Feature correlation heatmap reveals sensor relationships and potential redundancies",
        f"  7. Average subject contributes {analysis_results['subject_analysis']['avg_samples_per_subject']:.0f} samples across {analysis_results['subject_analysis']['avg_activities_per_subject']:.1f} activities"
    ]
    
    with open(f'{output_dir}/reports/enhanced_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Create CSV summaries for easy analysis
    # Subject summary
    subject_summary = pd.DataFrame([
        {
            'subject_id': subject_id,
            'total_samples': stats['total_samples'],
            'unique_activities': stats['unique_activities'],
            'files_count': stats['files_count']
        }
        for subject_id, stats in subject_stats.items()
    ])
    subject_summary.to_csv(f'{output_dir}/detailed_stats/subject_summary.csv', index=False)
    
    # Activity summary
    activity_summary = pd.DataFrame([
        {
            'activity': activity,
            'total_samples': stats['total_samples'],
            'subjects_count': stats['subjects_count'],
            'avg_samples_per_subject': stats['avg_samples_per_subject']
        }
        for activity, stats in activity_stats.items()
    ])
    activity_summary.to_csv(f'{output_dir}/detailed_stats/activity_summary.csv', index=False)
    
    # 11. Output summary
    print("\n✅ Enhanced analysis completed!")
    print("=" * 60)
    print(f"📊 Dataset: {analysis_results['dataset_info']['total_samples']:,} samples, {analysis_results['subject_analysis']['total_subjects']} subjects")
    print(f"🏷️  Activities: {analysis_results['activity_analysis']['total_activities']} types")
    print(f"📈 Data Quality: {analysis_results['data_quality']['completeness']:.1%} complete")
    print(f"🎯 High-quality Activities: {analysis_results['activity_analysis']['activities_with_1000_samples']} (≥1000 samples)")
    print(f"🔥 Feature Correlations: {correlation_stats['strong_correlations_count']} strong correlations found")
    print(f"📁 Results saved in: {output_dir}/ directory")
    print("   📊 Charts: plots/enhanced_analysis.png")
    print("   🔥 Feature Heatmaps: plots/feature_correlation_heatmap_70features.png")
    print("   🎨 Clustered Heatmap: plots/feature_correlation_heatmap_clustered.png")
    print("   🌳 Treemaps: treemaps/ folder (comprehensive treemap visualizations)")
    print("      📊 Activity Distribution: treemaps/activity_distribution_treemap.png")
    print("      🌐 Interactive Activity: treemaps/interactive_activity_treemap.html")
    print("      👥 Subject-Activity Hierarchical: treemaps/subject_activity_hierarchical_treemap.html")
    print("      🔧 Feature Groups: treemaps/feature_groups_treemap.png")
    print("      📋 Dataset Overview: treemaps/dataset_overview_treemap.html")
    print("   📄 Report: reports/enhanced_analysis_report.txt")
    print("   💾 Data: reports/enhanced_analysis_results.json")
    print("   🔗 Correlations: reports/feature_correlation_analysis.json")
    print("   📈 Details: detailed_stats/ folder (subject & activity statistics)")
    print("   📊 Correlation Matrix: detailed_stats/feature_correlation_matrix.csv")
    
    return analysis_results

# Alias for backward compatibility
def analyze_sonar_dataset(data_dir="../SONAR_ML", max_files=253, output_dir="statistics"):
    """Wrapper for enhanced analysis function"""
    return analyze_sonar_dataset_enhanced(data_dir, max_files, output_dir)

if __name__ == "__main__":
    # Run enhanced analysis
    results = analyze_sonar_dataset_enhanced()
    
    if results:
        print(f"\n🎉 Enhanced analysis completed successfully!")
        print(f"Found {results['activity_analysis']['total_activities']} activity categories from {results['subject_analysis']['total_subjects']} subjects")
        print(f"Total {results['dataset_info']['total_samples']:,} samples with detailed statistical analysis")
    else:
        print("\n❌ Analysis failed, please check data path") 