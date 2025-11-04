#!/usr/bin/env python3
"""Enhanced Time Series Analyzer for SONaR Dataset"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib for English display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedTimeAnalyzer:
    def __init__(self, data_dir="../SONAR_ML", output_dir="statistics"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        
    def extract_subject_id(self, filename):
        """Extract subject ID from filename"""
        try:
            if '_sub' in filename:
                return filename.split('_sub')[1].split('.')[0]
            return 'unknown'
        except:
            return 'unknown'
    
    def analyze_temporal_patterns(self, max_files=253):
        """Comprehensive temporal pattern analysis"""
        csv_files = glob.glob(os.path.join(self.data_dir, "*.csv"))[:max_files]
        results = {
            'files_analyzed': 0,
            'sampling_statistics': {},
            'activity_durations': {},
            'temporal_patterns': {},
            'subjects_temporal_stats': {}
        }
        
        print(f"🕐 Analyzing temporal patterns in {len(csv_files)} files...")
        
        all_sampling_rates = []
        all_activity_durations = []
        subject_patterns = {}
        
        for i, file_path in enumerate(csv_files):
            try:
                df = pd.read_csv(file_path)
                filename = os.path.basename(file_path)
                subject_id = self.extract_subject_id(filename)
                
                if 'SampleTimeFine' not in df.columns or len(df) < 10:
                    continue
                
                # Basic temporal analysis
                times = df['SampleTimeFine'].values
                time_intervals = np.diff(times)
                
                # Filter out outliers (intervals > 1000ms are likely data gaps)
                valid_intervals = time_intervals[time_intervals <= 1000]
                
                if len(valid_intervals) == 0:
                    continue
                
                # Calculate sampling statistics
                sampling_rate = 1000 / np.mean(valid_intervals) if np.mean(valid_intervals) > 0 else 0
                all_sampling_rates.append(sampling_rate)
                
                file_stats = {
                    'filename': filename,
                    'subject_id': subject_id,
                    'total_samples': len(df),
                    'duration_ms': float(times[-1] - times[0]),
                    'sampling_rate_hz': float(sampling_rate),
                    'mean_interval_ms': float(np.mean(valid_intervals)),
                    'std_interval_ms': float(np.std(valid_intervals)),
                    'min_interval_ms': float(np.min(valid_intervals)),
                    'max_interval_ms': float(np.max(valid_intervals)),
                    'interval_consistency': float(1 - (np.std(valid_intervals) / np.mean(valid_intervals))) if np.mean(valid_intervals) > 0 else 0
                }
                
                # Activity duration analysis
                if 'activity' in df.columns:
                    activity_changes = df['activity'].ne(df['activity'].shift()).cumsum()
                    activity_segments = df.groupby(activity_changes).agg({
                        'SampleTimeFine': ['first', 'last'],
                        'activity': 'first'
                    }).reset_index(drop=True)
                    
                    activity_segments.columns = ['start_time', 'end_time', 'activity']
                    activity_segments['duration_ms'] = activity_segments['end_time'] - activity_segments['start_time']
                    
                    # Filter valid activities
                    valid_activities = activity_segments[
                        (activity_segments['activity'] != 'null - activity') & 
                        (activity_segments['duration_ms'] > 100)  # At least 100ms duration
                    ]
                    
                    if len(valid_activities) > 0:
                        for _, activity_row in valid_activities.iterrows():
                            activity_name = activity_row['activity']
                            duration = activity_row['duration_ms']
                            all_activity_durations.append(duration)
                            
                            if activity_name not in results['activity_durations']:
                                results['activity_durations'][activity_name] = []
                            results['activity_durations'][activity_name].append(float(duration))
                        
                        file_stats['activities_count'] = len(valid_activities)
                        file_stats['unique_activities'] = valid_activities['activity'].nunique()
                        file_stats['avg_activity_duration_ms'] = float(valid_activities['duration_ms'].mean())
                        file_stats['std_activity_duration_ms'] = float(valid_activities['duration_ms'].std())
                
                # Store per-subject data
                if subject_id not in subject_patterns:
                    subject_patterns[subject_id] = []
                subject_patterns[subject_id].append(file_stats)
                
                results['files_analyzed'] += 1
                
                if i % 10 == 0:
                    print(f"  Processed: {i+1}/{len(csv_files)}")
                    
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
                continue
        
        # Calculate overall statistics
        if all_sampling_rates:
            results['sampling_statistics'] = {
                'mean_sampling_rate_hz': float(np.mean(all_sampling_rates)),
                'std_sampling_rate_hz': float(np.std(all_sampling_rates)),
                'min_sampling_rate_hz': float(np.min(all_sampling_rates)),
                'max_sampling_rate_hz': float(np.max(all_sampling_rates)),
                'median_sampling_rate_hz': float(np.median(all_sampling_rates)),
                'sampling_rate_consistency': float(1 - (np.std(all_sampling_rates) / np.mean(all_sampling_rates))) if np.mean(all_sampling_rates) > 0 else 0
            }
        
        # Calculate activity duration statistics
        for activity, durations in results['activity_durations'].items():
            if len(durations) > 0:
                results['activity_durations'][activity] = {
                    'count': len(durations),
                    'mean_duration_ms': float(np.mean(durations)),
                    'std_duration_ms': float(np.std(durations)),
                    'min_duration_ms': float(np.min(durations)),
                    'max_duration_ms': float(np.max(durations)),
                    'median_duration_ms': float(np.median(durations)),
                    'total_duration_ms': float(np.sum(durations))
                }
        
        # Calculate per-subject temporal statistics
        for subject_id, files_data in subject_patterns.items():
            if len(files_data) > 0:
                subject_sampling_rates = [f['sampling_rate_hz'] for f in files_data if 'sampling_rate_hz' in f]
                subject_durations = [f['duration_ms'] for f in files_data if 'duration_ms' in f]
                
                results['subjects_temporal_stats'][subject_id] = {
                    'files_count': len(files_data),
                    'total_samples': sum(f['total_samples'] for f in files_data),
                    'total_duration_ms': float(sum(subject_durations)) if subject_durations else 0,
                    'avg_sampling_rate_hz': float(np.mean(subject_sampling_rates)) if subject_sampling_rates else 0,
                    'std_sampling_rate_hz': float(np.std(subject_sampling_rates)) if len(subject_sampling_rates) > 1 else 0,
                    'sampling_consistency': float(1 - (np.std(subject_sampling_rates) / np.mean(subject_sampling_rates))) if len(subject_sampling_rates) > 1 and np.mean(subject_sampling_rates) > 0 else 1.0
                }
        
        # Generate temporal patterns summary
        if all_activity_durations:
            results['temporal_patterns'] = {
                'total_activity_instances': len(all_activity_durations),
                'mean_activity_duration_ms': float(np.mean(all_activity_durations)),
                'std_activity_duration_ms': float(np.std(all_activity_durations)),
                'short_activities_count': len([d for d in all_activity_durations if d < 5000]),  # < 5 seconds
                'medium_activities_count': len([d for d in all_activity_durations if 5000 <= d < 30000]),  # 5-30 seconds
                'long_activities_count': len([d for d in all_activity_durations if d >= 30000]),  # > 30 seconds
            }
        
        return results
    
    def generate_temporal_visualizations(self, analysis_results):
        """Generate comprehensive temporal visualizations"""
        if analysis_results['files_analyzed'] == 0:
            print("No data to visualize")
            return
        
        print("📊 Generating temporal visualizations...")
        
        # Create output directory
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Sampling rate distribution
        if analysis_results['subjects_temporal_stats']:
            plt.subplot(3, 3, 1)
            subject_rates = [stats['avg_sampling_rate_hz'] for stats in analysis_results['subjects_temporal_stats'].values()]
            plt.hist(subject_rates, bins=15, color='skyblue', edgecolor='black')
            plt.title('Sampling Rate Distribution by Subject', fontweight='bold')
            plt.xlabel('Sampling Rate (Hz)')
            plt.ylabel('Number of Subjects')
        
        # 2. Activity duration distribution
        if analysis_results['activity_durations']:
            plt.subplot(3, 3, 2)
            all_durations = []
            for activity_stats in analysis_results['activity_durations'].values():
                if isinstance(activity_stats, dict) and 'mean_duration_ms' in activity_stats:
                    all_durations.append(activity_stats['mean_duration_ms'])
            
            if all_durations:
                plt.hist(all_durations, bins=20, color='lightgreen', edgecolor='black')
                plt.title('Activity Duration Distribution', fontweight='bold')
                plt.xlabel('Duration (ms)')
                plt.ylabel('Number of Activities')
        
        # 3. Subject recording duration
        if analysis_results['subjects_temporal_stats']:
            plt.subplot(3, 3, 3)
            subject_durations = [stats['total_duration_ms']/60000 for stats in analysis_results['subjects_temporal_stats'].values()]  # Convert to minutes
            plt.bar(range(len(subject_durations)), sorted(subject_durations, reverse=True), color='orange')
            plt.title('Recording Duration by Subject', fontweight='bold')
            plt.xlabel('Subjects (Sorted)')
            plt.ylabel('Duration (minutes)')
        
        # 4. Sampling consistency
        if analysis_results['subjects_temporal_stats']:
            plt.subplot(3, 3, 4)
            consistencies = [stats['sampling_consistency'] for stats in analysis_results['subjects_temporal_stats'].values()]
            plt.hist(consistencies, bins=15, color='mediumpurple', edgecolor='black')
            plt.title('Sampling Consistency Distribution', fontweight='bold')
            plt.xlabel('Consistency Score (0-1)')
            plt.ylabel('Number of Subjects')
        
        # 5. Activity duration categories
        if 'temporal_patterns' in analysis_results and analysis_results['temporal_patterns']:
            plt.subplot(3, 3, 5)
            categories = ['Short\n(<5s)', 'Medium\n(5-30s)', 'Long\n(>30s)']
            counts = [
                analysis_results['temporal_patterns']['short_activities_count'],
                analysis_results['temporal_patterns']['medium_activities_count'],
                analysis_results['temporal_patterns']['long_activities_count']
            ]
            plt.pie(counts, labels=categories, autopct='%1.1f%%', colors=['lightcoral', 'gold', 'lightsteelblue'])
            plt.title('Activity Duration Categories', fontweight='bold')
        
        # 6. Top activities by average duration
        if analysis_results['activity_durations']:
            plt.subplot(3, 3, 6)
            activity_items = [(name, stats['mean_duration_ms']) for name, stats in analysis_results['activity_durations'].items() 
                             if isinstance(stats, dict) and 'mean_duration_ms' in stats]
            if activity_items:
                activity_items = sorted(activity_items, key=lambda x: x[1], reverse=True)[:10]
                activities, durations = zip(*activity_items)
                plt.barh(range(len(activities)), durations, color='salmon')
                plt.title('Top 10 Activities by Avg Duration', fontweight='bold')
                plt.xlabel('Duration (ms)')
                plt.yticks(range(len(activities)), activities)
        
        # 7. Subject sample count distribution
        if analysis_results['subjects_temporal_stats']:
            plt.subplot(3, 3, 7)
            sample_counts = [stats['total_samples'] for stats in analysis_results['subjects_temporal_stats'].values()]
            plt.boxplot(sample_counts)
            plt.title('Sample Count Distribution', fontweight='bold')
            plt.ylabel('Number of Samples')
            plt.xlabel('All Subjects')
        
        # 8. Sampling rate vs consistency
        if analysis_results['subjects_temporal_stats']:
            plt.subplot(3, 3, 8)
            rates = [stats['avg_sampling_rate_hz'] for stats in analysis_results['subjects_temporal_stats'].values()]
            consistencies = [stats['sampling_consistency'] for stats in analysis_results['subjects_temporal_stats'].values()]
            plt.scatter(rates, consistencies, alpha=0.6, color='teal')
            plt.title('Sampling Rate vs Consistency', fontweight='bold')
            plt.xlabel('Sampling Rate (Hz)')
            plt.ylabel('Consistency Score')
        
        # 9. Summary statistics
        plt.subplot(3, 3, 9)
        if 'sampling_statistics' in analysis_results and analysis_results['sampling_statistics']:
            summary_labels = ['Files', 'Subjects', 'Activities', 'Avg Rate']
            summary_values = [
                analysis_results['files_analyzed'],
                len(analysis_results['subjects_temporal_stats']),
                len(analysis_results['activity_durations']),
                analysis_results['sampling_statistics']['mean_sampling_rate_hz']
            ]
            plt.bar(summary_labels, summary_values, color='gold')
            plt.title('Temporal Analysis Summary', fontweight='bold')
            plt.ylabel('Count/Rate')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Temporal visualizations saved to {self.output_dir}/plots/temporal_analysis.png")
    
    def save_temporal_results(self, analysis_results):
        """Save detailed temporal analysis results"""
        os.makedirs(f"{self.output_dir}/reports", exist_ok=True)
        os.makedirs(f"{self.output_dir}/detailed_stats", exist_ok=True)
        
        # Save comprehensive results
        with open(f'{self.output_dir}/reports/temporal_analysis_results.json', 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False, default=str)
        
        # Generate text report
        report_lines = [
            "SONaR Dataset - Temporal Analysis Report",
            "=" * 50,
            f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "ANALYSIS OVERVIEW:",
            f"  Files Analyzed: {analysis_results['files_analyzed']}",
            f"  Subjects: {len(analysis_results['subjects_temporal_stats'])}",
            f"  Activities Analyzed: {len(analysis_results['activity_durations'])}",
            ""
        ]
        
        if 'sampling_statistics' in analysis_results and analysis_results['sampling_statistics']:
            report_lines.extend([
                "SAMPLING STATISTICS:",
                f"  Mean Sampling Rate: {analysis_results['sampling_statistics']['mean_sampling_rate_hz']:.1f} ± {analysis_results['sampling_statistics']['std_sampling_rate_hz']:.1f} Hz",
                f"  Sampling Rate Range: {analysis_results['sampling_statistics']['min_sampling_rate_hz']:.1f} - {analysis_results['sampling_statistics']['max_sampling_rate_hz']:.1f} Hz",
                f"  Sampling Consistency: {analysis_results['sampling_statistics']['sampling_rate_consistency']:.2%}",
                ""
            ])
        
        if 'temporal_patterns' in analysis_results and analysis_results['temporal_patterns']:
            report_lines.extend([
                "TEMPORAL PATTERNS:",
                f"  Total Activity Instances: {analysis_results['temporal_patterns']['total_activity_instances']}",
                f"  Average Activity Duration: {analysis_results['temporal_patterns']['mean_activity_duration_ms']:.0f} ± {analysis_results['temporal_patterns']['std_activity_duration_ms']:.0f} ms",
                f"  Short Activities (<5s): {analysis_results['temporal_patterns']['short_activities_count']}",
                f"  Medium Activities (5-30s): {analysis_results['temporal_patterns']['medium_activities_count']}",
                f"  Long Activities (>30s): {analysis_results['temporal_patterns']['long_activities_count']}",
                ""
            ])
        
        # Top activities by duration
        if analysis_results['activity_durations']:
            activity_items = [(name, stats['mean_duration_ms']) for name, stats in analysis_results['activity_durations'].items() 
                             if isinstance(stats, dict) and 'mean_duration_ms' in stats]
            if activity_items:
                activity_items = sorted(activity_items, key=lambda x: x[1], reverse=True)[:5]
                report_lines.extend([
                    "TOP 5 ACTIVITIES BY DURATION:",
                    *[f"  {i+1}. {name}: {duration:.0f} ms" for i, (name, duration) in enumerate(activity_items)],
                    ""
                ])
        
        # Subject statistics summary
        if analysis_results['subjects_temporal_stats']:
            total_duration_hours = sum(stats['total_duration_ms'] for stats in analysis_results['subjects_temporal_stats'].values()) / (1000 * 60 * 60)
            avg_consistency = np.mean([stats['sampling_consistency'] for stats in analysis_results['subjects_temporal_stats'].values()])
            
            report_lines.extend([
                "SUBJECT STATISTICS:",
                f"  Total Recording Time: {total_duration_hours:.1f} hours",
                f"  Average Sampling Consistency: {avg_consistency:.2%}",
                f"  Subjects with High Consistency (>90%): {sum(1 for stats in analysis_results['subjects_temporal_stats'].values() if stats['sampling_consistency'] > 0.9)}",
                ""
            ])
        
        with open(f'{self.output_dir}/reports/temporal_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        # Create CSV summaries
        if analysis_results['subjects_temporal_stats']:
            subject_df = pd.DataFrame([
                {
                    'subject_id': subject_id,
                    'files_count': stats['files_count'],
                    'total_samples': stats['total_samples'],
                    'total_duration_minutes': stats['total_duration_ms'] / 60000,
                    'avg_sampling_rate_hz': stats['avg_sampling_rate_hz'],
                    'sampling_consistency': stats['sampling_consistency']
                }
                for subject_id, stats in analysis_results['subjects_temporal_stats'].items()
            ])
            subject_df.to_csv(f'{self.output_dir}/detailed_stats/subject_temporal_summary.csv', index=False)
        
        if analysis_results['activity_durations']:
            activity_df = pd.DataFrame([
                {
                    'activity': activity,
                    'instances_count': stats['count'] if isinstance(stats, dict) else 0,
                    'mean_duration_ms': stats['mean_duration_ms'] if isinstance(stats, dict) else 0,
                    'std_duration_ms': stats['std_duration_ms'] if isinstance(stats, dict) else 0,
                    'total_duration_ms': stats['total_duration_ms'] if isinstance(stats, dict) else 0
                }
                for activity, stats in analysis_results['activity_durations'].items()
            ])
            activity_df.to_csv(f'{self.output_dir}/detailed_stats/activity_temporal_summary.csv', index=False)

def run_enhanced_time_analysis(data_dir="../SONAR_ML", max_files=253, output_dir="statistics"):
    """Run comprehensive temporal analysis"""
    analyzer = EnhancedTimeAnalyzer(data_dir, output_dir)
    
    print("🕐 Starting enhanced temporal analysis...")
    results = analyzer.analyze_temporal_patterns(max_files)
    
    if results['files_analyzed'] > 0:
        analyzer.generate_temporal_visualizations(results)
        analyzer.save_temporal_results(results)
        
        print(f"\n✅ Enhanced temporal analysis completed!")
        print(f"📊 Analyzed {results['files_analyzed']} files from {len(results['subjects_temporal_stats'])} subjects")
        if 'sampling_statistics' in results and results['sampling_statistics']:
            print(f"⏱️  Average sampling rate: {results['sampling_statistics']['mean_sampling_rate_hz']:.1f} Hz")
        print(f"🎯 Found {len(results['activity_durations'])} activities with temporal patterns")
        print(f"📁 Results saved in: {output_dir}/ directory")
        
        return results
    else:
        print("❌ No valid temporal data found")
        return None

# Backward compatibility function
def run_time_analysis():
    """Simple time analysis function for backward compatibility"""
    analyzer = EnhancedTimeAnalyzer()
    result = analyzer.analyze_temporal_patterns(max_files=10)
    if result and result['files_analyzed'] > 0:
        if 'sampling_statistics' in result and result['sampling_statistics']:
            print(f"✓ Time analysis: {result['files_analyzed']} files, {result['sampling_statistics']['mean_sampling_rate_hz']:.1f} Hz")
        return result
    return None

if __name__ == "__main__":
    # Run enhanced temporal analysis
    run_enhanced_time_analysis() 