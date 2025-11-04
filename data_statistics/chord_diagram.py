#!/usr/bin/env python3
"""
Beautiful Activity Chord Diagram Generator
使用Plotly创建美观的护理活动相关性弦图

This script creates beautiful, publication-ready chord diagrams for activity correlations
with advanced styling, color gradients, and image export capabilities.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import glob
import os
from collections import defaultdict, Counter
from itertools import combinations
import json
import warnings
warnings.filterwarnings('ignore')

# Configure plotly for high-quality static exports
pio.kaleido.scope.mathjax = None

class BeautifulChordDiagramGenerator:
    """
    美观弦图生成器
    Creates beautiful chord diagrams with advanced styling and export capabilities
    """
    
    def __init__(self, data_path="../SONAR_ML/*.csv", max_files=None):
        """
        初始化生成器
        
        Args:
            data_path: Path pattern to CSV files
            max_files: Maximum number of files to load (None for all files)
        """
        self.data_path = data_path
        self.max_files = max_files
        self.data = None
        self.activities = None
        self.activity_counts = None
        
        # Beautiful color palettes
        self.color_palettes = {
            'medical': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                       '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9'],
            'professional': ['#2C3E50', '#3498DB', '#E74C3C', '#9B59B6', '#1ABC9C',
                           '#F39C12', '#34495E', '#E67E22', '#95A5A6', '#16A085'],
            'nature': ['#27AE60', '#2ECC71', '#3498DB', '#9B59B6', '#E74C3C',
                      '#F39C12', '#1ABC9C', '#E67E22', '#34495E', '#95A5A6'],
            'pastel': ['#FFB6C1', '#87CEEB', '#98FB98', '#DDA0DD', '#F0E68C',
                      '#FFE4B5', '#AFEEEE', '#D3D3D3', '#FFDAB9', '#E0E0E0'],
            'vibrant': ['#FF1744', '#FF9100', '#FFEA00', '#00E676', '#00BCD4',
                       '#3F51B5', '#9C27B0', '#E91E63', '#FF5722', '#607D8B']
        }
        
        print("🎨 Beautiful Chord Diagram Generator")
        print("="*50)
        
    def load_data(self):
        """加载SONAR数据"""
        print("📂 Loading SONAR data...")
        
        csv_files = glob.glob(self.data_path)
        files_to_load = csv_files if self.max_files is None else csv_files[:self.max_files]
        
        if self.max_files is None:
            print(f"Found {len(csv_files)} CSV files, loading ALL files")
        else:
            print(f"Found {len(csv_files)} CSV files, loading up to {self.max_files}")
        
        all_data = []
        loaded_count = 0
        
        for file_path in files_to_load:
            try:
                df = pd.read_csv(file_path)
                
                if 'activity' not in df.columns:
                    continue
                    
                # Filter out null activities
                df = df[df['activity'] != 'null - activity']
                df = df[df['activity'].notna()]
                
                if len(df) > 100:  # Only keep files with substantial data
                    filename = os.path.basename(file_path)
                    df['file_id'] = filename
                    all_data.append(df)
                    loaded_count += 1
                    
            except Exception as e:
                print(f"Warning: Error loading {file_path}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No valid data could be loaded!")
        
        self.data = pd.concat(all_data, ignore_index=True)
        print(f"✅ Loaded {loaded_count} files with {len(self.data):,} total samples")
        
        # Get activity statistics
        self.activity_counts = self.data['activity'].value_counts()
        self.activities = self.activity_counts.index.tolist()
        
        print(f"📊 Found {len(self.activities)} unique activities:")
        for i, (activity, count) in enumerate(self.activity_counts.head(10).items()):
            print(f"  {i+1:2d}. {activity}: {count:,} samples")
        
        if len(self.activities) > 10:
            print(f"  ... and {len(self.activities) - 10} more activities")
        
        return self.data
    
    def compute_cooccurrence_matrix(self, window_size=30, min_activity_count=1000):
        """
        计算活动共现矩阵
        """
        print(f"\n🔍 Computing co-occurrence matrix (window_size={window_size})...")
        
        # Filter activities with sufficient samples
        frequent_activities = self.activity_counts[self.activity_counts >= min_activity_count].index.tolist()
        print(f"Using {len(frequent_activities)} activities with ≥{min_activity_count} samples")
        
        # Initialize co-occurrence matrix
        n_activities = len(frequent_activities)
        cooccurrence_matrix = np.zeros((n_activities, n_activities))
        activity_to_idx = {activity: idx for idx, activity in enumerate(frequent_activities)}
        
        # Group by file to process sequences
        for file_id, group in self.data.groupby('file_id'):
            activities_sequence = group['activity'].tolist()
            
            # Sliding window approach
            for i in range(len(activities_sequence) - window_size + 1):
                window_activities = activities_sequence[i:i + window_size]
                
                # Count co-occurrences within this window
                window_frequent = [act for act in window_activities if act in activity_to_idx]
                
                for act1, act2 in combinations(set(window_frequent), 2):
                    idx1, idx2 = activity_to_idx[act1], activity_to_idx[act2]
                    cooccurrence_matrix[idx1, idx2] += 1
                    cooccurrence_matrix[idx2, idx1] += 1  # Symmetric
        
        print(f"✅ Co-occurrence matrix computed: {cooccurrence_matrix.shape}")
        return cooccurrence_matrix, frequent_activities
    
    def create_chord_diagram(self, matrix, activities, title="Nursing Activity Correlation Chord Diagram", 
                                     style='medical', save_image=True, output_dir="chord_outputs"):
        """
        Create beautiful chord diagram for nursing activities
        
        Args:
            matrix: Correlation matrix
            activities: List of activity names
            title: Plot title (in English)
            style: Color style ('medical', 'professional', 'nature', 'pastel', 'vibrant')
            save_image: Whether to save as image file
            output_dir: Directory to save outputs
        """
        print(f"\n🎨 Creating beautiful chord diagram: {title}")
        print(f"🎭 Using {style} color scheme")
        
        # Normalize matrix
        matrix_norm = matrix / (matrix.max() + 1e-8) if matrix.max() > 0 else matrix
        
        # Create figure with high DPI for quality
        fig = go.Figure()
        
        n_activities = len(activities)
        if n_activities < 2:
            print("Warning: Need at least 2 activities for chord diagram")
            return fig
        
        # Get color palette
        colors = self.color_palettes.get(style, self.color_palettes['medical'])
        if len(activities) > len(colors):
            colors = colors * (len(activities) // len(colors) + 1)
        colors = colors[:len(activities)]
        
        # Position activities on circle with enhanced spacing
        angles = [2 * np.pi * i / n_activities for i in range(n_activities)]
        outer_radius = 1.2
        inner_radius = 0.95
        
        # Activity positions
        x_outer = [outer_radius * np.cos(angle) for angle in angles]
        y_outer = [outer_radius * np.sin(angle) for angle in angles]
        x_inner = [inner_radius * np.cos(angle) for angle in angles]
        y_inner = [inner_radius * np.sin(angle) for angle in angles]
        
        # Calculate connection strengths for better threshold
        connections = []
        for i in range(n_activities):
            for j in range(i + 1, n_activities):
                if matrix_norm[i, j] > 0:
                    connections.append(matrix_norm[i, j])
        
        if not connections:
            print("Warning: No connections found")
            return fig
        
        # Lower threshold to show more connections - use 5th percentile to show most correlations
        threshold = np.percentile(connections, 5) if len(connections) > 10 else 0.01 * np.mean(connections)
        
        # Draw enhanced connections (chords) with improved smoothness
        for i in range(n_activities):
            for j in range(i + 1, n_activities):
                if matrix_norm[i, j] > threshold:
                    # Calculate correlation strength relative to maximum
                    correlation_strength = matrix_norm[i, j]
                    
                    # Create much smoother bezier curves
                    x0, y0 = x_inner[i], y_inner[i]
                    x1, y1 = x_inner[j], y_inner[j]
                    
                    # Improved control points for smoother curves
                    # Make control points deeper for more pronounced curves
                    control_depth = 0.1 + 0.6 * correlation_strength
                    cx0, cy0 = x0 * control_depth, y0 * control_depth
                    cx1, cy1 = x1 * control_depth, y1 * control_depth
                    
                    # Create very smooth curve with many more points
                    t = np.linspace(0, 1, 200)  # Increased from 50 to 200 points for ultra-smooth curves
                    x_curve = (1-t)**3 * x0 + 3*(1-t)**2*t * cx0 + 3*(1-t)*t**2 * cx1 + t**3 * x1
                    y_curve = (1-t)**3 * y0 + 3*(1-t)**2*t * cy0 + 3*(1-t)*t**2 * cy1 + t**3 * y1
                    
                    # Improved opacity mapping - use quadratic function for better visibility
                    # Map correlation strength (0-1) to opacity (0.2-0.9) with quadratic scaling
                    opacity = 0.2 + 0.7 * (correlation_strength ** 0.7)
                    
                    # Line width also reflects correlation strength
                    line_width = 0.5 + 6 * correlation_strength
                    
                    fig.add_trace(go.Scatter(
                        x=x_curve, y=y_curve,
                        mode='lines',
                        line=dict(color=colors[i], width=line_width),
                        opacity=opacity,
                        hovertemplate=f"<b>{activities[i]}</b> ↔ <b>{activities[j]}</b><br>" +
                                    f"Correlation Strength: {matrix[i,j]:.3f}<br>" +
                                    f"Normalized Strength: {correlation_strength:.3f}<br>" +
                                    f"Opacity: {opacity:.2f}<extra></extra>",
                        showlegend=False
                    ))
        
        # Create beautiful activity arcs
        for i, (activity, color) in enumerate(zip(activities, colors)):
            angle = angles[i]
            
            # Calculate arc for each activity - smaller spacing between classes
            arc_span = 2 * np.pi / n_activities * 0.95  # 95% of available space for smaller gaps
            start_angle = angle - arc_span / 2
            end_angle = angle + arc_span / 2
            
            # Create arc points
            arc_angles = np.linspace(start_angle, end_angle, 30)
            x_arc_outer = [outer_radius * np.cos(a) for a in arc_angles]
            y_arc_outer = [outer_radius * np.sin(a) for a in arc_angles]
            x_arc_inner = [inner_radius * np.cos(a) for a in arc_angles]
            y_arc_inner = [inner_radius * np.sin(a) for a in arc_angles]
            
            # Create filled arc
            x_arc = x_arc_outer + x_arc_inner[::-1] + [x_arc_outer[0]]
            y_arc = y_arc_outer + y_arc_inner[::-1] + [y_arc_outer[0]]
            
            # Activity count for sizing
            activity_count = self.activity_counts[activity] if hasattr(self, 'activity_counts') else 1000
            
            fig.add_trace(go.Scatter(
                x=x_arc, y=y_arc,
                fill='toself',
                fillcolor=color,
                mode='lines',
                line=dict(color='white', width=2),
                opacity=0.8,
                hovertemplate=f"<b>{activity}</b><br>" +
                            f"Sample Count: {activity_count:,}<br>" +
                            f"Percentage: {activity_count/len(self.data)*100:.1f}%<extra></extra>",
                showlegend=False,
                name=f"arc_{i}"
            ))
        
        # Add activity labels with enhanced positioning
        label_radius = 1.35
        for i, activity in enumerate(activities):
            angle = angles[i]
            x_label = label_radius * np.cos(angle)
            y_label = label_radius * np.sin(angle)
            
            # Enhanced label positioning
            if np.cos(angle) > 0:
                text_anchor = 'left'
            else:
                text_anchor = 'right'
            
            # Truncate long names
            display_name = activity[:25] + "..." if len(activity) > 25 else activity
            
            fig.add_annotation(
                x=x_label, y=y_label,
                text=f"<b>{display_name}</b>",
                showarrow=False,
                font=dict(size=11, color='#2C3E50', family="Arial Black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=colors[i],
                borderwidth=1,
                borderpad=4,
                align=text_anchor
            )
        
        # Enhanced layout with professional styling
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(size=20, color='#2C3E50', family="Arial Black"),
                pad=dict(t=20)
            ),
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False, 
                range=[-1.8, 1.8],
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False, 
                range=[-1.8, 1.8]
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1200,
            height=1200,
            margin=dict(l=80, r=80, t=100, b=80),
            showlegend=False,
            font=dict(family="Arial", size=12)
        )
        
        # Add subtitle with statistics
        fig.add_annotation(
            text=f"Analyzed {len(activities)} nursing activities | Threshold: {threshold:.3f} | Connections: {len([c for c in connections if c > threshold])}",
            xref="paper", yref="paper",
            x=0.5, y=0.02,
            showarrow=False,
            font=dict(size=12, color='#7F8C8D'),
            bgcolor="rgba(255,255,255,0.9)"
        )
        
        # Save as high-quality image if requested
        if save_image:
            os.makedirs(output_dir, exist_ok=True)
            
            # Clean title for filename
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            
            # Save as PNG with high DPI (professional format only)
            png_path = os.path.join(output_dir, f"{safe_title}_{style}.png")
            fig.write_image(png_path, width=1200, height=1200, scale=2)
            print(f"  💾 Saved high-quality PNG: {png_path}")
        
        return fig
    
    def create_network_style_diagram(self, matrix, activities, title="Network Style Activity Diagram",
                                   style='professional', save_image=True, output_dir="chord_outputs"):
        """
        创建网络风格的活动关系图
        """
        print(f"\n🕸️ Creating network style diagram: {title}")
        
        # Normalize matrix
        matrix_norm = matrix / (matrix.max() + 1e-8) if matrix.max() > 0 else matrix
        
        fig = go.Figure()
        
        n_activities = len(activities)
        if n_activities < 2:
            return fig
        
        # Use force-directed layout simulation (simplified)
        # Position nodes in a circular layout with some randomization
        angles = [2 * np.pi * i / n_activities + np.random.normal(0, 0.1) for i in range(n_activities)]
        radii = [0.8 + 0.4 * np.random.random() for _ in range(n_activities)]
        
        x_pos = [r * np.cos(a) for r, a in zip(radii, angles)]
        y_pos = [r * np.sin(a) for r, a in zip(radii, angles)]
        
        # Get colors
        colors = self.color_palettes.get(style, self.color_palettes['professional'])
        if len(activities) > len(colors):
            colors = colors * (len(activities) // len(colors) + 1)
        colors = colors[:len(activities)]
        
        # Draw connections
        threshold = np.percentile(matrix_norm[matrix_norm > 0], 65) if np.sum(matrix_norm > 0) > 0 else 0
        
        edge_trace = []
        for i in range(n_activities):
            for j in range(i + 1, n_activities):
                if matrix_norm[i, j] > threshold:
                    weight = matrix_norm[i, j]
                    
                    fig.add_trace(go.Scatter(
                        x=[x_pos[i], x_pos[j]], 
                        y=[y_pos[i], y_pos[j]],
                        mode='lines',
                        line=dict(
                            color=colors[i],
                            width=1 + 8 * weight
                        ),
                        opacity=0.5 + 0.4 * weight,
                        hovertemplate=f"{activities[i]} ↔ {activities[j]}<br>强度: {matrix[i,j]:.3f}<extra></extra>",
                        showlegend=False
                    ))
        
        # Draw nodes
        node_sizes = [20 + 30 * (self.activity_counts[act] / self.activity_counts.max()) 
                     if hasattr(self, 'activity_counts') else 25 for act in activities]
        
        fig.add_trace(go.Scatter(
            x=x_pos, y=y_pos,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=colors,
                line=dict(width=3, color='white'),
                opacity=0.9
            ),
            text=[act[:15] + "..." if len(act) > 15 else act for act in activities],
            textposition='middle center',
            textfont=dict(size=9, color='white', family="Arial Black"),
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=18, color='#2C3E50')),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            width=1000,
            height=1000,
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        # Save if requested
        if save_image:
            os.makedirs(output_dir, exist_ok=True)
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip().replace(' ', '_')
            
            png_path = os.path.join(output_dir, f"{safe_title}_network_{style}.png")
            fig.write_image(png_path, width=1000, height=1000, scale=2)
            print(f"  💾 Saved network PNG: {png_path}")
        
        return fig
    
    def compute_feature_correlation_matrix(self, sample_size=50000):
        """
        计算70个传感器特征之间的相关性矩阵
        
        Args:
            sample_size: 用于计算相关性的样本数量（避免内存不足）
        """
        print(f"\n🔬 Computing feature correlation matrix...")
        
        if self.data is None:
            self.load_data()
        
        # Get all numeric feature columns (exclude 'activity', 'file_id', 'SampleTimeFine')
        exclude_cols = ['activity', 'file_id', 'SampleTimeFine']
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        print(f"Found {len(feature_cols)} sensor features")
        
        # Sample data to avoid memory issues
        if len(self.data) > sample_size:
            sample_data = self.data.sample(n=sample_size, random_state=42)
            print(f"Sampling {sample_size:,} rows from {len(self.data):,} total samples")
        else:
            sample_data = self.data
            print(f"Using all {len(sample_data):,} samples")
        
        # Select only numeric features and handle missing values
        feature_data = sample_data[feature_cols].select_dtypes(include=[np.number])
        feature_data = feature_data.fillna(feature_data.mean())
        
        print(f"Processing {feature_data.shape[1]} numeric features")
        
        # Compute correlation matrix
        correlation_matrix = feature_data.corr().values
        feature_names = feature_data.columns.tolist()
        
        # Replace NaN with 0
        correlation_matrix = np.nan_to_num(correlation_matrix)
        
        print(f"✅ Feature correlation matrix computed: {correlation_matrix.shape}")
        return correlation_matrix, feature_names
    
    def create_feature_chord_diagram(self, correlation_matrix, feature_names, 
                                   title="Sensor Feature Correlation Chord Diagram",
                                   style='professional', min_correlation=0.3, 
                                   save_image=True, output_dir="chord_outputs"):
        """
        Create sensor feature correlation chord diagram
        
        Args:
            correlation_matrix: Feature correlation matrix
            feature_names: List of feature names
            title: Plot title (in English)
            style: Color style
            min_correlation: Minimum correlation threshold for display (lowered to 0.3)
            save_image: Whether to save as image
            output_dir: Output directory
        """
        print(f"\n🎨 Creating feature correlation chord diagram: {title}")
        print(f"🎭 Using {style} color scheme, min correlation: {min_correlation}")
        
        # Get absolute correlation values for filtering
        abs_corr_matrix = np.abs(correlation_matrix)
        
        # Create figure
        fig = go.Figure()
        
        n_features = len(feature_names)
        if n_features < 2:
            print("Warning: Need at least 2 features for chord diagram")
            return fig
        
        # Create expanded color palette for many features
        base_colors = self.color_palettes.get(style, self.color_palettes['professional'])
        colors = []
        for i in range(n_features):
            colors.append(base_colors[i % len(base_colors)])
        
        # Position features on circle
        angles = [2 * np.pi * i / n_features for i in range(n_features)]
        outer_radius = 1.2
        inner_radius = 0.95
        
        # Feature positions
        x_inner = [inner_radius * np.cos(angle) for angle in angles]
        y_inner = [inner_radius * np.sin(angle) for angle in angles]
        
        # Find significant correlations
        significant_correlations = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if abs_corr_matrix[i, j] >= min_correlation:
                    significant_correlations.append((i, j, correlation_matrix[i, j]))
        
        print(f"Found {len(significant_correlations)} significant correlations (≥{min_correlation})")
        
        if not significant_correlations:
            print("Warning: No significant correlations found")
            return fig
        
        # Draw enhanced feature correlation connections with ribbon style (same as activity diagram)
        for i, j, corr_value in significant_correlations:
            # Calculate correlation strength
            abs_corr = abs(corr_value)
            
            # Calculate feature arc parameters (same approach as activity diagram)
            arc_span_i = 2 * np.pi / n_features * 0.8
            arc_span_j = 2 * np.pi / n_features * 0.8
            
            # Calculate connection width based on correlation strength
            # Use a portion of the arc span proportional to correlation strength
            connection_width_i = arc_span_i * (0.1 + 0.4 * abs_corr)
            connection_width_j = arc_span_j * (0.1 + 0.4 * abs_corr)
            
            # Start and end angles for the connection spans
            angle_i = angles[i]
            angle_j = angles[j]
            
            # Calculate start span on arc i
            start_angle_i1 = angle_i - connection_width_i / 2
            start_angle_i2 = angle_i + connection_width_i / 2
            
            # Calculate end span on arc j
            end_angle_j1 = angle_j - connection_width_j / 2
            end_angle_j2 = angle_j + connection_width_j / 2
            
            # Points on the inner edge of arcs
            x_start_1 = inner_radius * np.cos(start_angle_i1)
            y_start_1 = inner_radius * np.sin(start_angle_i1)
            x_start_2 = inner_radius * np.cos(start_angle_i2)
            y_start_2 = inner_radius * np.sin(start_angle_i2)
            
            x_end_1 = inner_radius * np.cos(end_angle_j1)
            y_end_1 = inner_radius * np.sin(end_angle_j1)
            x_end_2 = inner_radius * np.cos(end_angle_j2)
            y_end_2 = inner_radius * np.sin(end_angle_j2)
            
            # Create control points for smoother curves
            control_depth = 0.15 + 0.5 * abs_corr
            
            # Control points for the bezier curves
            cx_start_1 = x_start_1 * control_depth
            cy_start_1 = y_start_1 * control_depth
            cx_start_2 = x_start_2 * control_depth
            cy_start_2 = y_start_2 * control_depth
            
            cx_end_1 = x_end_1 * control_depth
            cy_end_1 = y_end_1 * control_depth
            cx_end_2 = x_end_2 * control_depth
            cy_end_2 = y_end_2 * control_depth
            
            # Create smooth curves with more points for the chord ribbon
            t = np.linspace(0, 1, 100)
            
            # Upper curve (from start_1 to end_1)
            x_curve_upper = (1-t)**3 * x_start_1 + 3*(1-t)**2*t * cx_start_1 + 3*(1-t)*t**2 * cx_end_1 + t**3 * x_end_1
            y_curve_upper = (1-t)**3 * y_start_1 + 3*(1-t)**2*t * cy_start_1 + 3*(1-t)*t**2 * cy_end_1 + t**3 * y_end_1
            
            # Lower curve (from start_2 to end_2)
            x_curve_lower = (1-t)**3 * x_start_2 + 3*(1-t)**2*t * cx_start_2 + 3*(1-t)*t**2 * cx_end_2 + t**3 * x_end_2
            y_curve_lower = (1-t)**3 * y_start_2 + 3*(1-t)**2*t * cy_start_2 + 3*(1-t)*t**2 * cy_end_2 + t**3 * y_end_2
            
            # Create connecting arcs at both ends for more realistic ribbon appearance
            # Start arc (connecting the two start points)
            start_arc_angles = np.linspace(start_angle_i1, start_angle_i2, 10)
            x_start_arc = [inner_radius * np.cos(a) for a in start_arc_angles]
            y_start_arc = [inner_radius * np.sin(a) for a in start_arc_angles]
            
            # End arc (connecting the two end points)
            end_arc_angles = np.linspace(end_angle_j1, end_angle_j2, 10)
            x_end_arc = [inner_radius * np.cos(a) for a in end_arc_angles]
            y_end_arc = [inner_radius * np.sin(a) for a in end_arc_angles]
            
            # Create the complete ribbon by combining all segments
            # Start arc -> upper curve -> end arc -> lower curve (reversed) -> back to start
            x_ribbon = np.concatenate([
                x_start_arc, 
                x_curve_upper[1:], 
                x_end_arc[::-1], 
                x_curve_lower[::-1][1:], 
                [x_start_arc[0]]
            ])
            y_ribbon = np.concatenate([
                y_start_arc, 
                y_curve_upper[1:], 
                y_end_arc[::-1], 
                y_curve_lower[::-1][1:], 
                [y_start_arc[0]]
            ])
            
            # Improved opacity mapping - use quadratic function for better correlation visibility
            # Map correlation strength (0-1) to opacity (0.25-0.95) with enhanced scaling
            opacity = 0.25 + 0.7 * (abs_corr ** 0.6)
            
            # Color based on positive/negative correlation
            fill_color = colors[i] if corr_value > 0 else '#E74C3C'
            
            # Create the filled chord ribbon with enhanced visual appeal
            fig.add_trace(go.Scatter(
                x=x_ribbon, y=y_ribbon,
                fill='toself',
                fillcolor=fill_color,
                mode='lines',
                line=dict(color=fill_color, width=1),  # Subtle outline
                opacity=opacity,
                hovertemplate=f"<b>{feature_names[i]}</b> ↔ <b>{feature_names[j]}</b><br>" +
                            f"Correlation Coefficient: {corr_value:.3f}<br>" +
                            f"Type: {'Positive' if corr_value > 0 else 'Negative'}<br>" +
                            f"Connection Width: {connection_width_i*180/np.pi:.1f}°<br>" +
                            f"Opacity: {opacity:.2f}<extra></extra>",
                showlegend=False
            ))
            
            # Add gradient effect for strong correlations (same as activity diagram)
            if abs_corr > 0.7:  # Only for very strong correlations
                # Create a slightly narrower ribbon for the gradient overlay
                narrow_width_i = connection_width_i * 0.7
                narrow_width_j = connection_width_j * 0.7
                
                narrow_start_i1 = angle_i - narrow_width_i / 2
                narrow_start_i2 = angle_i + narrow_width_i / 2
                narrow_end_j1 = angle_j - narrow_width_j / 2
                narrow_end_j2 = angle_j + narrow_width_j / 2
                
                # Calculate narrow ribbon points
                x_narrow_start_1 = inner_radius * np.cos(narrow_start_i1)
                y_narrow_start_1 = inner_radius * np.sin(narrow_start_i1)
                x_narrow_start_2 = inner_radius * np.cos(narrow_start_i2)
                y_narrow_start_2 = inner_radius * np.sin(narrow_start_i2)
                
                x_narrow_end_1 = inner_radius * np.cos(narrow_end_j1)
                y_narrow_end_1 = inner_radius * np.sin(narrow_end_j1)
                x_narrow_end_2 = inner_radius * np.cos(narrow_end_j2)
                y_narrow_end_2 = inner_radius * np.sin(narrow_end_j2)
                
                # Create narrow curves
                cx_narrow_start_1 = x_narrow_start_1 * control_depth
                cy_narrow_start_1 = y_narrow_start_1 * control_depth
                cx_narrow_end_1 = x_narrow_end_1 * control_depth
                cy_narrow_end_1 = y_narrow_end_1 * control_depth
                
                cx_narrow_start_2 = x_narrow_start_2 * control_depth
                cy_narrow_start_2 = y_narrow_start_2 * control_depth
                cx_narrow_end_2 = x_narrow_end_2 * control_depth
                cy_narrow_end_2 = y_narrow_end_2 * control_depth
                
                # Narrow curves
                x_narrow_upper = (1-t)**3 * x_narrow_start_1 + 3*(1-t)**2*t * cx_narrow_start_1 + 3*(1-t)*t**2 * cx_narrow_end_1 + t**3 * x_narrow_end_1
                y_narrow_upper = (1-t)**3 * y_narrow_start_1 + 3*(1-t)**2*t * cy_narrow_start_1 + 3*(1-t)*t**2 * cy_narrow_end_1 + t**3 * y_narrow_end_1
                
                x_narrow_lower = (1-t)**3 * x_narrow_start_2 + 3*(1-t)**2*t * cx_narrow_start_2 + 3*(1-t)*t**2 * cx_narrow_end_2 + t**3 * x_narrow_end_2
                y_narrow_lower = (1-t)**3 * y_narrow_start_2 + 3*(1-t)**2*t * cy_narrow_start_2 + 3*(1-t)*t**2 * cy_narrow_end_2 + t**3 * y_narrow_end_2
                
                # Create narrow connecting arcs
                narrow_start_arc_angles = np.linspace(narrow_start_i1, narrow_start_i2, 8)
                x_narrow_start_arc = [inner_radius * np.cos(a) for a in narrow_start_arc_angles]
                y_narrow_start_arc = [inner_radius * np.sin(a) for a in narrow_start_arc_angles]
                
                narrow_end_arc_angles = np.linspace(narrow_end_j1, narrow_end_j2, 8)
                x_narrow_end_arc = [inner_radius * np.cos(a) for a in narrow_end_arc_angles]
                y_narrow_end_arc = [inner_radius * np.sin(a) for a in narrow_end_arc_angles]
                
                # Combine narrow ribbon
                x_narrow_ribbon = np.concatenate([
                    x_narrow_start_arc, 
                    x_narrow_upper[1:], 
                    x_narrow_end_arc[::-1], 
                    x_narrow_lower[::-1][1:], 
                    [x_narrow_start_arc[0]]
                ])
                y_narrow_ribbon = np.concatenate([
                    y_narrow_start_arc, 
                    y_narrow_upper[1:], 
                    y_narrow_end_arc[::-1], 
                    y_narrow_lower[::-1][1:], 
                    [y_narrow_start_arc[0]]
                ])
                
                # Add gradient overlay with lighter color
                gradient_color = colors[j] if corr_value > 0 else '#FF8A80'
                fig.add_trace(go.Scatter(
                    x=x_narrow_ribbon, y=y_narrow_ribbon,
                    fill='toself',
                    fillcolor=gradient_color,
                    mode='lines',
                    line=dict(color=gradient_color, width=0),
                    opacity=opacity * 0.4,  # More transparent
                    hoverinfo='skip',
                    showlegend=False
                ))
        
        # Create feature arcs (smaller for many features)
        for i, (feature, color) in enumerate(zip(feature_names, colors)):
            angle = angles[i]
            
            # Smaller arc span for many features
            arc_span = 2 * np.pi / n_features * 0.8
            start_angle = angle - arc_span / 2
            end_angle = angle + arc_span / 2
            
            # Create arc points
            arc_angles = np.linspace(start_angle, end_angle, 15)
            x_arc_outer = [outer_radius * np.cos(a) for a in arc_angles]
            y_arc_outer = [outer_radius * np.sin(a) for a in arc_angles]
            x_arc_inner = [inner_radius * np.cos(a) for a in arc_angles]
            y_arc_inner = [inner_radius * np.sin(a) for a in arc_angles]
            
            # Create filled arc
            x_arc = x_arc_outer + x_arc_inner[::-1] + [x_arc_outer[0]]
            y_arc = y_arc_outer + y_arc_inner[::-1] + [y_arc_outer[0]]
            
            fig.add_trace(go.Scatter(
                x=x_arc, y=y_arc,
                fill='toself',
                fillcolor=color,
                mode='lines',
                line=dict(color='white', width=1),
                opacity=0.7,
                hovertemplate=f"<b>{feature}</b><br>Sensor Feature<extra></extra>",
                showlegend=False,
                name=f"feature_{i}"
            ))
        
        # Add feature labels with radial arrangement (same format as activity diagram)
        for i, feature in enumerate(feature_names):
            angle = angles[i]
            
            # Split feature name and arrange radially (same as activity diagram)
            words = feature.split('_')
            if len(words) > 1:
                # For multi-part features, arrange them radially from inner to outer
                for j, word in enumerate(words):
                    # Position each word at increasing radial distances
                    word_radius = outer_radius + 0.15 + j * 0.12
                    x_word = word_radius * np.cos(angle)
                    y_word = word_radius * np.sin(angle)
                    
                    # Font size decreases from inner to outer (same as activity diagram)
                    font_size = max(6, 9 - j)
                    # Opacity decreases from inner to outer for depth effect
                    opacity = max(0.4, 1.0 - j * 0.15)
                    
                    # Only show labels for subset to avoid overcrowding (show ~30 features)
                    if i % max(1, n_features // 30) == 0:
                        fig.add_trace(go.Scatter(
                            x=[x_word], y=[y_word],
                            mode='text',
                            text=[word],
                            textposition='middle center',
                            textfont=dict(
                                size=font_size, 
                                color='#2C3E50',
                                family="Arial"
                            ),
                            opacity=opacity,
                            hoverinfo='skip',
                            showlegend=False
                        ))
            else:
                # For single-word features, use default positioning
                label_radius = outer_radius + 0.2
                x_label = label_radius * np.cos(angle)
                y_label = label_radius * np.sin(angle)
                
                # Only show labels for subset to avoid overcrowding
                if i % max(1, n_features // 30) == 0:
                    fig.add_trace(go.Scatter(
                        x=[x_label], y=[y_label],
                        mode='text',
                        text=[feature],
                        textposition='middle center',
                        textfont=dict(size=8, color='#2C3E50', family="Arial"),
                        hoverinfo='skip',
                        showlegend=False
                    ))
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=f"<b>{title}</b>",
                x=0.5,
                font=dict(size=18, color='#2C3E50', family="Arial Black"),
                pad=dict(t=20)
            ),
            xaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False, 
                range=[-1.8, 1.8],
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                showgrid=False, 
                zeroline=False, 
                showticklabels=False, 
                range=[-1.8, 1.8]
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            width=1400,
            height=1400,
            margin=dict(l=100, r=100, t=120, b=100),
            showlegend=False,
            font=dict(family="Arial", size=10)
        )
        
        # Add subtitle
        fig.add_annotation(
            text=f"Analyzed {n_features} sensor features | Threshold: {min_correlation} | Significant Correlations: {len(significant_correlations)}",
            xref="paper", yref="paper",
            x=0.5, y=0.02,
            showarrow=False,
            font=dict(size=12, color='#7F8C8D'),
            bgcolor="rgba(255,255,255,0.9)"
        )
        
        # Save image
        if save_image:
            os.makedirs(output_dir, exist_ok=True)
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            
            png_path = os.path.join(output_dir, f"{safe_title}_{style}.png")
            fig.write_image(png_path, width=1400, height=1400, scale=2)
            print(f"  💾 Saved feature correlation chord diagram: {png_path}")
        
        return fig
    
    def run_beautiful_analysis(self, min_activity_count=500, window_size=30, 
                             styles=['medical', 'professional'], save_all=True):
        """
        运行美观的弦图分析
        
        Args:
            min_activity_count: Minimum samples for activity inclusion
            window_size: Window size for co-occurrence
            styles: List of color styles to generate
            save_all: Whether to save all outputs
        """
        print("\n" + "="*60)
        print("🎨 RUNNING BEAUTIFUL CHORD DIAGRAM ANALYSIS")
        print("="*60)
        
        # Load data
        if self.data is None:
            self.load_data()
        
        # Create output directory
        output_dir = "chord_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute correlation matrix
        cooccur_matrix, cooccur_activities = self.compute_cooccurrence_matrix(
            window_size=window_size, min_activity_count=min_activity_count
        )
        
        figures = {}
        
        # Generate chord diagrams in professional style only
        for style in styles:
            print(f"\n🎭 Generating {style} style chord diagram...")
            
            # Beautiful chord diagram only
            chord_fig = self.create_chord_diagram(
                cooccur_matrix, cooccur_activities,
                title=f"Nursing Activity Correlation Chord Diagram - {style.title()} Style",
                style=style,
                save_image=save_all,
                output_dir=output_dir
            )
            figures[f'chord_{style}'] = chord_fig
        
        # Save analysis summary
        if save_all:
            summary = {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'total_samples': len(self.data),
                'total_activities': len(self.activities),
                'analyzed_activities': len(cooccur_activities),
                'min_activity_count': min_activity_count,
                'window_size': window_size,
                'styles_generated': styles,
                'top_activities': self.activity_counts.head(10).to_dict(),
                'matrix_stats': {
                    'max_correlation': float(cooccur_matrix.max()),
                    'mean_correlation': float(cooccur_matrix.mean()),
                    'non_zero_connections': int(np.sum(cooccur_matrix > 0))
                }
            }
            
            summary_path = os.path.join(output_dir, "analysis_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"  📋 Saved analysis summary: {summary_path}")
        
        print("\n" + "="*60)
        print("🎉 BEAUTIFUL CHORD DIAGRAM ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📊 Generated {len(figures)} beautiful visualizations")
        print(f"💾 All outputs saved to: {output_dir}/")
        print(f"🌈 Styles created: {', '.join(styles)}")
        print(f"📈 Activities analyzed: {len(cooccur_activities)}")
        
        return figures, cooccur_matrix, cooccur_activities
    
    def run_feature_correlation_analysis(self, sample_size=50000, min_correlation=0.3, 
                                       styles=['professional'], save_all=True):
        """
        运行传感器特征相关性分析
        
        Args:
            sample_size: Sample size for correlation computation
            min_correlation: Minimum correlation threshold for display
            styles: List of color styles to generate
            save_all: Whether to save all outputs
        """
        print("\n" + "="*60)
        print("🔬 RUNNING FEATURE CORRELATION ANALYSIS")
        print("="*60)
        
        # Compute feature correlation matrix
        correlation_matrix, feature_names = self.compute_feature_correlation_matrix(sample_size=sample_size)
        
        # Create output directory
        output_dir = "chord_outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        figures = {}
        
        # Generate feature correlation chord diagrams
        for style in styles:
            print(f"\n🎭 Generating {style} style feature correlation chord diagram...")
            
            feature_fig = self.create_feature_chord_diagram(
                correlation_matrix, feature_names,
                title=f"Sensor Feature Correlation Chord Diagram - {style.title()} Style",
                style=style,
                min_correlation=min_correlation,
                save_image=save_all,
                output_dir=output_dir
            )
            figures[f'feature_chord_{style}'] = feature_fig
        
        # Save correlation matrix and feature analysis
        if save_all:
            # Save correlation matrix as CSV
            corr_df = pd.DataFrame(correlation_matrix, 
                                 index=feature_names, 
                                 columns=feature_names)
            corr_csv_path = os.path.join(output_dir, "feature_correlation_matrix.csv")
            corr_df.to_csv(corr_csv_path)
            print(f"  📊 Saved correlation matrix: {corr_csv_path}")
            
            # Create analysis summary
            abs_corr_matrix = np.abs(correlation_matrix)
            np.fill_diagonal(abs_corr_matrix, 0)  # Remove self-correlations
            
            high_corr_pairs = []
            for i in range(len(feature_names)):
                for j in range(i + 1, len(feature_names)):
                    if abs_corr_matrix[i, j] >= min_correlation:
                        high_corr_pairs.append({
                            'feature_1': feature_names[i],
                            'feature_2': feature_names[j],
                            'correlation': correlation_matrix[i, j],
                            'abs_correlation': abs_corr_matrix[i, j]
                        })
            
            # Sort by absolute correlation
            high_corr_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
            
            summary = {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'total_features': len(feature_names),
                'sample_size': sample_size,
                'min_correlation_threshold': min_correlation,
                'significant_correlations': len(high_corr_pairs),
                'max_correlation': float(abs_corr_matrix.max()),
                'mean_abs_correlation': float(np.mean(abs_corr_matrix[abs_corr_matrix > 0])),
                'feature_names': feature_names,
                'top_correlations': high_corr_pairs[:20],  # Top 20 correlations
                'styles_generated': styles
            }
            
            summary_path = os.path.join(output_dir, "feature_correlation_analysis.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"  📋 Saved feature analysis summary: {summary_path}")
        
        print("\n" + "="*60)
        print("🎉 FEATURE CORRELATION ANALYSIS COMPLETE!")
        print("="*60)
        print(f"📊 Generated {len(figures)} feature correlation visualizations")
        print(f"💾 All outputs saved to: {output_dir}/")
        print(f"🔬 Features analyzed: {len(feature_names)}")
        print(f"🔗 Significant correlations found: {len(high_corr_pairs) if 'high_corr_pairs' in locals() else 'N/A'}")
        
        return figures, correlation_matrix, feature_names


def main():
    """主函数 - 运行弦图分析（活动相关性和特征相关性）"""
    
    print("🎵 NURSING ACTIVITY CHORD DIAGRAM ANALYSIS SUITE")
    print("="*70)
    
    # Create generator - USE ALL FILES to include all 22 activities and all features
    generator = BeautifulChordDiagramGenerator(data_path="../SONAR_ML/*.csv", max_files=None)
    
    # Option 1: Run activity correlation analysis
    print("\n🏥 PART 1: ACTIVITY CORRELATION ANALYSIS")
    print("-" * 50)
    activity_figures, activity_matrix, activities = generator.run_beautiful_analysis(
        min_activity_count=100,  # Lowered to include more activities
        window_size=30,
        styles=['professional'],  # Only professional style
        save_all=True
    )
    
    # Option 2: Run feature correlation analysis
    print("\n🔬 PART 2: SENSOR FEATURE CORRELATION ANALYSIS")
    print("-" * 50)
    feature_figures, feature_matrix, feature_names = generator.run_feature_correlation_analysis(
        sample_size=50000,  # Sample size for correlation computation
        min_correlation=0.5,  # Only show correlations ≥ 0.5
        styles=['professional'],
        save_all=True
    )
    
    # Summary of results
    print("\n" + "="*70)
    print("🎉 COMPLETE CHORD DIAGRAM ANALYSIS FINISHED!")
    print("="*70)
    print(f"📊 Total visualizations created: {len(activity_figures) + len(feature_figures)}")
    print(f"🏥 Activity chord diagrams: {len(activity_figures)}")
    print(f"🔬 Feature correlation chord diagrams: {len(feature_figures)}")
    print(f"💾 All outputs saved to: chord_outputs/")
    print(f"\n📁 Generated files:")
    print(f"  🏥 Activity analysis:")
    print(f"    - Activity chord diagram (PNG)")
    print(f"    - Activity analysis summary (JSON)")
    print(f"  🔬 Feature analysis:")
    print(f"    - Feature correlation chord diagram (PNG)")
    print(f"    - Feature correlation matrix (CSV)")
    print(f"    - Feature correlation analysis (JSON)")
    
    return {
        'activity_figures': activity_figures,
        'feature_figures': feature_figures,
        'activity_matrix': activity_matrix,
        'feature_matrix': feature_matrix,
        'activities': activities,
        'features': feature_names
    }


if __name__ == "__main__":
    results = main() 