import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

def plot_metrics_history():
    """Plot metrics history from metrics_history.json"""
    # Load metrics history
    metrics_file = Path('experiments/metrics/metrics_history.json')
    if not metrics_file.exists():
        print("No metrics history file found")
        return
        
    with open(metrics_file) as f:
        experiments = json.load(f)
        
    # Convert to DataFrame
    data = []
    for exp in experiments:
        if exp['git']['status'] == 'pending':
            continue
            
        row = {
            'timestamp': datetime.fromisoformat(exp['timestamp']),
            'commit': exp['git']['commit_hash'][:8],
            'commit_message': exp['git']['commit_message']
        }
        
        # Extract metrics
        metrics = exp['metrics_by_dataset']['test']
        for metric, value in metrics.items():
            if isinstance(value, dict):
                # Handle nested metrics (e.g., precision by class)
                for k, v in value.items():
                    row[f'{metric}_{k}'] = v
            else:
                row[metric] = value
                
        data.append(row)
        
    df = pd.DataFrame(data)
    if df.empty:
        print("No valid metrics data found")
        return
        
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Model Metrics Over Time', fontsize=16)
    
    # Plot accuracy
    ax = axes[0, 0]
    ax.plot(df['timestamp'], df['accuracy'], marker='o')
    ax.set_title('Accuracy')
    ax.set_ylim(0, 1)
    ax.grid(True)
    
    # Plot precision by class
    ax = axes[0, 1]
    precision_cols = [col for col in df.columns if col.startswith('precision_')]
    for col in precision_cols:
        ax.plot(df['timestamp'], df[col], marker='o', label=col.split('_')[1])
    ax.set_title('Precision by Class')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True)
    
    # Plot training time
    ax = axes[1, 0]
    ax.plot(df['timestamp'], df['training_time'], marker='o')
    ax.set_title('Training Time (seconds)')
    ax.grid(True)
    
    # Plot inference time
    ax = axes[1, 1]
    ax.plot(df['timestamp'], df['inference_time'], marker='o')
    ax.set_title('Inference Time (seconds)')
    ax.grid(True)
    
    # Add commit labels
    for ax in axes.flat:
        for i, commit in enumerate(df['commit']):
            ax.annotate(commit, 
                       (df['timestamp'].iloc[i], df['accuracy'].iloc[i]),
                       xytext=(0, 10), textcoords='offset points',
                       rotation=45, fontsize=8)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('output/visualization')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'metrics_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    
if __name__ == '__main__':
    plot_metrics_history() 