import streamlit as st
import os
import json
import numpy as np
from PIL import Image
import logging
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import traceback
from typing import Dict, Optional, List, Any
from dataclasses import dataclass
from git import Repo
import networkx as nx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DiffInfo:
    file_path: str
    change_type: str
    lines_changed: int
    additions: int
    deletions: int

class GitAnalyzer:
    def __init__(self, repo_path: str):
        self.repo = Repo(repo_path)
    
    def get_all_commits(self) -> List[Dict]:
        """Get all commits from the repository"""
        commits = []
        for commit in self.repo.iter_commits():
            commits.append({
                'hash': commit.hexsha,
                'message': commit.message,
                'author': commit.author.name,
                'timestamp': datetime.fromtimestamp(commit.committed_datetime.timestamp())
            })
        return commits
    
    def get_diff(self, commit_hash: str) -> List[DiffInfo]:
        """Get diff information for a specific commit"""
        try:
            commit = self.repo.commit(commit_hash)
            if not commit.parents:
                return []
            
            parent = commit.parents[0]
            diffs = []
            
            for diff in commit.diff(parent):
                diffs.append(DiffInfo(
                    file_path=diff.a_path if diff.a_path else diff.b_path,
                    change_type=diff.change_type,
                    lines_changed=diff.additions + diff.deletions,
                    additions=diff.additions,
                    deletions=diff.deletions
                ))
            
            return diffs
        except Exception as e:
            logger.error(f"Error getting diff for commit {commit_hash}: {e}")
            return []

class DiffAnalyzer:
    def analyze(self, diff_info: List[DiffInfo]) -> List[Dict]:
        """Analyze diff information to determine structural changes"""
        analysis = []
        for diff in diff_info:
            analysis.append({
                'file': diff.file_path,
                'structural': {
                    'change_type': diff.change_type,
                    'lines_changed': diff.lines_changed,
                    'additions': diff.additions,
                    'deletions': diff.deletions
                }
            })
        return analysis

@dataclass
class NodeData:
    analysis: List[Dict]
    metrics: Optional[Dict] = None

class ModelGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.metrics_history = {}
    
    def add_node(self, commit_hash: str, analysis: List[Dict], metrics: Optional[Dict] = None):
        """Add a node to the graph with its associated data"""
        self.graph.add_node(commit_hash, data=NodeData(analysis=analysis, metrics=metrics))
        if metrics:
            self.metrics_history[commit_hash] = metrics

# Constants
PREPROCESSED_DATA_DIR = 'preprocessed_data'
METADATA_FILE = os.path.join(PREPROCESSED_DATA_DIR, 'metadata.json')
METRICS_FILE = os.getenv('METRICS_FILE', 'experiments/metrics/metrics_history.json')
DEMO_REPO_PATH = os.getenv('MAEE_REPO_PATH', '/Users/erph/Documents/maee-demo-repo')

def check_environment():
    """Check if all required files and directories exist."""
    try:
        if not os.path.exists(PREPROCESSED_DATA_DIR):
            raise FileNotFoundError(f"Preprocessed data directory not found: {PREPROCESSED_DATA_DIR}")
        if not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
        if not os.path.exists(DEMO_REPO_PATH):
            raise FileNotFoundError(f"Demo repo path not found: {DEMO_REPO_PATH}")
        return True
    except Exception as e:
        st.error(f"Environment check failed: {str(e)}")
        return False

def load_config():
    """Load MAEE configuration"""
    repo_path = DEMO_REPO_PATH  # Use the constant defined at the top
    
    config_path = Path(repo_path) / '.maee' / 'config.json'
    if not config_path.exists():
        st.error(f"Repository not initialized at {repo_path}. Please run 'maee init <repo_path>' first")
        st.stop()
    
    with open(config_path) as f:
        return json.load(f)

def load_metadata():
    """Load metadata from preprocessed data"""
    try:
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading metadata: {e}")
        return None

def create_commit_graph(commits, metrics_history):
    """Create a graph showing commits and their relationships."""
    # Filter commits to only those with metrics and sort chronologically
    commits_with_metrics = []
    for commit in commits:
        commit_metrics = next(
            (m for m in metrics_history if m["git"]["commit_hash"] == commit['hash']),
            None
        )
        if commit_metrics:
            commits_with_metrics.append((commit, commit_metrics))
    
    # Sort commits by timestamp (oldest first)
    commits_with_metrics.sort(key=lambda x: x[0]['timestamp'])
    
    # Create nodes with chronological ordering
    nodes = []
    for i, (commit, metrics) in enumerate(commits_with_metrics):
        # Calculate average metrics across datasets
        avg_metrics = {}
        for dataset, dataset_metrics in metrics["metrics_by_dataset"].items():
            for metric, value in dataset_metrics.items():
                if metric not in avg_metrics:
                    avg_metrics[metric] = []
                avg_metrics[metric].append(value)
        
        avg_metrics = {k: sum(v)/len(v) for k, v in avg_metrics.items()}
        
        # Create hover text with commit info and metrics
        hover_text = f"Commit: {commit['hash'][:8]}<br>"
        hover_text += f"Message: {commit['message'].splitlines()[0]}<br>"
        hover_text += f"Date: {commit['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
        
        # Add detailed metrics to hover text
        hover_text += "<br>Metrics:<br>"
        for dataset, dataset_metrics in metrics["metrics_by_dataset"].items():
            hover_text += f"{dataset}:<br>"
            for metric, value in dataset_metrics.items():
                hover_text += f"  {metric}: {value:.4f}<br>"
        
        # Add changes summary
        if 'diff_analysis' in commit:
            changes = []
            for diff in commit['diff_analysis']:
                changes.append(f"{diff['file']} ({diff['structural']['change_type']})")
            if changes:
                hover_text += "<br>Changes:<br>"
                for change in changes[:3]:  # Show first 3 changes
                    hover_text += f"  {change}<br>"
                if len(changes) > 3:
                    hover_text += f"  ... and {len(changes) - 3} more"
        
        # Create node with metrics as y-value
        nodes.append({
            'x': [i],
            'y': [avg_metrics.get('accuracy', 0)],  # Use accuracy as y-value
            'text': [hover_text],
            'mode': 'markers+text',
            'marker': {
                'size': 15,
                'color': 'blue',
                'line': {'width': 2, 'color': 'black'}
            },
            'text': [f"{avg_metrics.get('accuracy', 0):.3f}"],  # Show accuracy on graph
            'textposition': 'top center',
            'name': commit['hash'][:8],
            'showlegend': False
        })
    
    # Create edges between consecutive commits
    edges = []
    for i in range(len(commits_with_metrics) - 1):
        edges.append({
            'x': [i, i + 1],
            'y': [
                nodes[i]['y'][0],
                nodes[i + 1]['y'][0]
            ],
            'mode': 'lines',
            'line': {'width': 2, 'color': 'gray'},
            'showlegend': False
        })
    
    # Create the figure
    fig = go.Figure()
    
    # Add edges first (so they appear behind nodes)
    for edge in edges:
        fig.add_trace(go.Scatter(**edge))
    
    # Add nodes
    for node in nodes:
        fig.add_trace(go.Scatter(**node))
    
    # Update layout
    fig.update_layout(
        title="Model Performance Over Time",
        showlegend=False,
        hovermode='closest',
        xaxis=dict(
            title="Commits (chronological)",
            ticktext=[node['name'] for node in nodes],
            tickvals=list(range(len(nodes))),
            tickangle=45,
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            title="Accuracy",
            showgrid=True,
            zeroline=False,
            range=[0, 1],
            gridcolor='lightgray'
        ),
        height=400,
        margin=dict(l=20, r=20, t=40, b=100)
    )
    
    return fig

def create_metrics_history_graph(commits_with_metrics, metrics_history):
    """Create a graph showing all metrics over time."""
    # Sort commits by timestamp (oldest first)
    commits_with_metrics.sort(key=lambda x: x['timestamp'])
    
    # Create a flattened DataFrame for visualization
    metrics_data = []
    for i, commit in enumerate(commits_with_metrics):
        # Get metrics for this commit
        commit_metrics = metrics_history.get(commit['hash'])
        if not commit_metrics:
            continue
            
        for dataset, dataset_metrics in commit_metrics.items():
            for metric, value in dataset_metrics.items():
                metrics_data.append({
                    'commit': commit['hash'][:8],
                    'commit_idx': i,
                    'dataset': dataset,
                    'metric': metric,
                    'value': value
                })
    
    if not metrics_data:
        return None

    metrics_df = pd.DataFrame(metrics_data)
    
    # Create a figure for each metric
    figures = []
    for metric in sorted(metrics_df['metric'].unique()):
        fig = go.Figure()
        
        # Add a trace for each dataset
        for dataset in sorted(metrics_df['dataset'].unique()):
            data = metrics_df[(metrics_df['metric'] == metric) & (metrics_df['dataset'] == dataset)]
            if not data.empty:
                fig.add_trace(go.Scatter(
                    x=data['commit_idx'],
                    y=data['value'],
                    name=dataset,
                    mode='lines+markers',
                    line=dict(width=2),
                    marker=dict(size=8),
                    hovertemplate=(
                        f"<b>Dataset:</b> {dataset}<br>" +
                        f"<b>Commit:</b> %{{customdata}}<br>" +
                        f"<b>Value:</b> %{{y:.4f}}<br>" +
                        "<extra></extra>"
                    ),
                    customdata=data['commit']
                ))
        
        # Update layout for this metric's figure
        fig.update_layout(
            title=f"{metric} Over Time",
            xaxis=dict(
                title="Commits (chronological)",
                ticktext=[commit['hash'][:8] for commit in commits_with_metrics],
                tickvals=list(range(len(commits_with_metrics))),
                tickangle=45,
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title=metric,
                showgrid=True,
                zeroline=False,
                range=[0, 1],
                gridcolor='lightgray'
            ),
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            height=300,
            margin=dict(l=20, r=20, t=40, b=100)
        )
        
        figures.append(fig)
    
    return figures

def display_examples(metadata):
    """Display MNIST examples in a grid"""
    if not metadata:
        st.error("No metadata available")
        return

    st.header("MNIST Examples")
    
    # Create two columns for regular and adversarial examples
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Regular MNIST")
        if 'regular_examples' in metadata:
            for idx, example in enumerate(metadata['regular_examples']):
                img_path = os.path.join(PREPROCESSED_DATA_DIR, example['image_path'])
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, caption=f"True: {example['true_label']}, Predicted: {example['pred_label']}", use_container_width=True)
    
    with col2:
        st.subheader("Adversarial MNIST")
        if 'adversarial_examples' in metadata:
            for idx, example in enumerate(metadata['adversarial_examples']):
                img_path = os.path.join(PREPROCESSED_DATA_DIR, example['image_path'])
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, caption=f"True: {example['true_label']}, Predicted: {example['pred_label']}", use_container_width=True)

def main():
    try:
        # Check environment first
        if not check_environment():
            st.error("Environment check failed. Please check the logs for details.")
            return
            
        st.title("MAEE Dashboard")
        
        # Load configuration
        config = load_config()
        repo_path = config['repo_path']
        
        # Initialize analyzers
        git_analyzer = GitAnalyzer(repo_path)
        diff_analyzer = DiffAnalyzer()
        
        # Initialize the model graph
        model_graph = ModelGraph()
        
        # Load metrics history
        metrics_path = Path(repo_path) / 'experiments' / 'metrics' / 'metrics_history.json'
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics_history = json.load(f)
                    for entry in metrics_history:
                        if 'git' in entry and 'metrics_by_dataset' in entry:
                            commit_hash = entry['git']['commit_hash']
                            metrics = entry['metrics_by_dataset']
                            model_graph.metrics_history[commit_hash] = metrics
            except json.JSONDecodeError:
                st.error(f"Error reading metrics file: {metrics_path}. File may be corrupted.")
            except Exception as e:
                st.error(f"Unexpected error reading metrics file: {str(e)}")
        else:
            st.info("No metrics history found. Run some evaluations to generate metrics.")
        
        # Get all commits from the repository
        all_commits = git_analyzer.get_all_commits()
        
        # Filter commits that have metrics
        commits_with_metrics = [commit for commit in all_commits if commit['hash'] in model_graph.metrics_history]
        
        # Add pending metrics if they exist
        pending_metrics = next((entry for entry in metrics_history if entry.get('git', {}).get('status') == 'pending'), None)
        if pending_metrics:
            pending_commit = {
                'hash': 'pending',
                'timestamp': datetime.now(),
                'message': 'Pending Evaluation'
            }
            commits_with_metrics.append(pending_commit)
        
        if not commits_with_metrics:
            st.warning("No commits with metrics found. Please run evaluation first.")
            return
        
        # Sort commits by date (oldest first)
        commits_with_metrics.sort(key=lambda x: x['timestamp'])
        
        # Add commits to the model graph
        for commit in commits_with_metrics:
            metrics = model_graph.metrics_history.get(commit['hash'])
            if commit['hash'] == 'pending':
                # For pending commits, we don't have diff info
                analysis = []
            else:
                # For committed changes, get the diff info
                diff_info = git_analyzer.get_diff(commit['hash'])
                analysis = diff_analyzer.analyze(diff_info)
            
            model_graph.add_node(commit['hash'], analysis, metrics)
        
        # Display commit graph
        st.subheader("Commit Analysis Graph")
        commit_graph = create_commit_graph(all_commits, metrics_history)
        if commit_graph is not None:
            st.plotly_chart(commit_graph, use_container_width=True, key="commit_graph")
        else:
            st.info("No commit graph available - need commits with metrics to generate visualization.")
        
        # Display metrics history
        if model_graph.metrics_history and commits_with_metrics:
            st.subheader("Metrics History")
            metrics_history_graphs = create_metrics_history_graph(commits_with_metrics, model_graph.metrics_history)
            if metrics_history_graphs is not None:
                for i, fig in enumerate(metrics_history_graphs):
                    st.plotly_chart(fig, use_container_width=True, key=f"metric_history_{i}")
            else:
                st.info("No metrics history available to generate visualization.")
        
        # Display commits with metrics
        st.subheader("Commits with Metrics")
        
        if not commits_with_metrics:
            st.info("No commits found with tracked metrics.")
        else:
            for i, commit in enumerate(commits_with_metrics):
                with st.expander(f"Commit {commit['hash'][:8]} - {commit['message'].split()[0]}"):
                    st.write(f"Author: {commit['author']}")
                    st.write(f"Date: {commit['timestamp']}")
                    st.write(f"Message: {commit['message']}")
                    
                    # Display metrics if available
                    node_data = model_graph.graph.nodes[commit['hash']]['data']
                    if node_data.metrics:
                        # Create a DataFrame for metrics
                        metrics_data = []
                        for dataset, dataset_metrics in node_data.metrics.items():
                            for metric, value in dataset_metrics.items():
                                metrics_data.append({
                                    'Dataset': dataset,
                                    'Metric': metric,
                                    'Value': f"{value:.4f}"
                                })
                        
                        metrics_df = pd.DataFrame(metrics_data)
                        metrics_df = metrics_df.pivot(index='Metric', columns='Dataset', values='Value')
                        st.dataframe(metrics_df, use_container_width=True)
                    
                    # Display commit changes
                    st.write("\nChanges:")
                    if commit['hash'] == 'pending':
                        st.write("No changes to display - metrics from current training session")
                    else:
                        try:
                            diff_info = git_analyzer.get_diff(commit['hash'])
                            if diff_info:
                                for diff in diff_info:
                                    st.write(f"- {diff.file_path} ({diff.change_type})")
                                    if diff.lines_changed > 0:
                                        st.write(f"  Lines changed: {diff.lines_changed}")
                            else:
                                st.write("No changes found in this commit")
                        except Exception as e:
                            st.error(f"Error getting diff info: {str(e)}")
        
        # Display MNIST Examples
        st.subheader("MNIST Model Analysis")
        
        # Load metadata
        metadata = load_metadata()
        
        # Display examples
        display_examples(metadata)
            
    except Exception as e:
        st.error(f"Critical error in main: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        st.error(f"Traceback: {traceback.format_exc()}")
        logger.error(f"Critical error: {str(e)}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main() 