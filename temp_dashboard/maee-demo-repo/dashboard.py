import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
import numpy as np
import os
import sys
import networkx as nx
from scipy import stats
import plotly.figure_factory as ff

# Constants
METRICS_FILE = os.getenv('METRICS_FILE', 'metrics_history.json')
GITHUB_REPO = "multiagenteval/maee-demo-repo"  # Update this with your actual GitHub repo

# Model overview
MODEL_OVERVIEW = {
    "architecture": {
        "name": "Baseline CNN with Residual Connections",
        "description": """
        A convolutional neural network designed for MNIST digit classification with the following key features:
        
        - **Input Layer**: 1x28x28 (grayscale MNIST images)
        - **Convolutional Blocks**: 
            - Two main blocks with residual connections
            - Each block contains: Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU
            - Skip connections preserve gradient flow
        - **Hidden Dimensions**: [32, 64] channels
        - **Dropout Rate**: 0.1 (for regularization)
        - **Output Layer**: 10 units (one for each digit)
        
        The model uses residual connections to improve gradient flow and feature preservation through the network.
        """,
        "training": """
        - **Optimizer**: Adam
        - **Learning Rate**: 0.001
        - **Batch Size**: 32
        - **Training Data**: MNIST training set
        - **Adversarial Training**: FGSM with ε=0.3
        """
    },
    "metrics": {
        "accuracy": """
        The proportion of correctly classified digits out of all predictions.
        - Range: 0.0 to 1.0
        - Higher is better
        - Most intuitive metric for overall performance
        """,
        "f1": """
        Harmonic mean of precision and recall.
        - Range: 0.0 to 1.0
        - Higher is better
        - Better than accuracy when classes are imbalanced
        - Penalizes both false positives and false negatives
        """,
        "precision": """
        The proportion of true positives out of all positive predictions.
        - Range: 0.0 to 1.0
        - Higher is better
        - Measures how many of the predicted positives are actually positive
        - Important when false positives are costly
        """,
        "recall": """
        The proportion of true positives out of all actual positives.
        - Range: 0.0 to 1.0
        - Higher is better
        - Measures how many of the actual positives are correctly identified
        - Important when false negatives are costly
        """
    }
}

# Dataset descriptions
DATASET_DESCRIPTIONS = {
    "test": {
        "name": "Standard Test Set",
        "description": "The standard MNIST test set containing 10,000 images. This is the primary evaluation dataset that provides a baseline measure of model performance on clean, unmodified data.",
        "interpretation": "High performance on this dataset indicates good basic recognition capabilities. However, high performance here alone doesn't guarantee robustness.",
        "use_case": "Baseline performance evaluation"
    },
    "test_balanced": {
        "name": "Balanced Test Set",
        "description": "A class-balanced version of the MNIST test set, ensuring equal representation of all digits. This helps identify any class imbalance issues in the model's predictions.",
        "interpretation": "Performance on this dataset reveals if the model has biases towards certain classes. A significant difference from the standard test set indicates class imbalance issues.",
        "use_case": "Class balance analysis"
    },
    "test_adversarial": {
        "name": "Adversarial Test Set",
        "description": "MNIST test set with FGSM (Fast Gradient Sign Method) adversarial attacks applied. This tests the model's robustness against carefully crafted perturbations.",
        "interpretation": "Performance here indicates the model's vulnerability to adversarial attacks. Lower performance suggests the model is more susceptible to adversarial examples.",
        "use_case": "Robustness evaluation"
    }
}

def load_data(file_path):
    """Load and process the metrics data with proper error handling"""
    try:
        # Try absolute path first
        if os.path.isabs(file_path):
            full_path = file_path
        else:
            # Try relative to current directory
            full_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_path)
        
        if not os.path.exists(full_path):
            st.error(f"Metrics file not found at: {full_path}")
            st.info("Please ensure the metrics file is in the correct location or set the METRICS_FILE environment variable.")
            return pd.DataFrame()
            
        with open(full_path, 'r') as f:
            data = json.load(f)
        
        # Process the data into a more usable format
        processed_data = []
        for entry in data:
            timestamp = datetime.fromisoformat(entry['timestamp'])
            commit_hash = entry['git']['commit_hash']
            commit_msg = entry['git']['commit_message'].split('\n')[0]  # First line
            
            for dataset, metrics in entry['metrics_by_dataset'].items():
                metrics_row = {
                    'timestamp': timestamp,
                    'commit_hash': commit_hash,
                    'commit_message': commit_msg,
                    'dataset': dataset,
                    **metrics
                }
                processed_data.append(metrics_row)
        
        return pd.DataFrame(processed_data)
    except Exception as e:
        st.error(f"Error loading metrics data: {str(e)}")
        return pd.DataFrame()

def calculate_aggregate_score(metrics):
    """Calculate a weighted aggregate score across all metrics"""
    weights = {
        'accuracy': 0.4,
        'f1': 0.3,
        'precision': 0.15,
        'recall': 0.15
    }
    return sum(metrics[metric] * weight for metric, weight in weights.items())

def get_github_commit_url(commit_hash):
    """Generate GitHub commit URL"""
    return f"https://github.com/{GITHUB_REPO}/commit/{commit_hash}"

def create_network_graph(df, selected_dataset):
    """Create an interactive network graph showing commit relationships and metric impacts"""
    G = nx.DiGraph()
    
    # Add nodes for each commit
    for idx, row in df.iterrows():
        G.add_node(row['commit_hash'][:7], 
                  metrics=row[['accuracy', 'f1', 'precision', 'recall']].to_dict(),
                  message=row['commit_message'],
                  timestamp=row['timestamp'])
    
    # Add edges between consecutive commits
    for i in range(len(df)-1):
        current = df.iloc[i]
        next_commit = df.iloc[i+1]
        G.add_edge(current['commit_hash'][:7], next_commit['commit_hash'][:7])
    
    # Convert to Plotly format
    pos = nx.spring_layout(G)
    edge_trace = go.Scatter(
        x=[], y=[],
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_trace = go.Scatter(
        x=[], y=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlOrRd',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Metric Impact',
                xanchor='left',
                titleside='right'
            )
        ),
        text=[],
        textposition="top center",
    )
    
    # Add edges
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    # Add nodes
    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([f"Commit: {node}<br>Message: {G.nodes[node]['message']}"])
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Commit Network and Metric Impact',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       annotations=[dict(
                           text="",
                           showarrow=False,
                           xref="paper", yref="paper",
                           x=0, y=0)],
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    
    return fig

def create_metric_correlation_heatmap(df):
    """Create a heatmap showing correlations between metrics"""
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    corr_matrix = df[metrics].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=metrics,
        y=metrics,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Metric Correlations',
        xaxis_title='Metrics',
        yaxis_title='Metrics',
        height=400
    )
    
    return fig

def create_radar_chart(df, selected_dataset):
    """Create a radar chart for comprehensive metric comparison"""
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    latest_metrics = df[metrics].iloc[-1]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=latest_metrics.values,
        theta=metrics,
        fill='toself',
        name='Latest Metrics'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title='Comprehensive Metric Analysis'
    )
    
    return fig

def create_enhanced_timeline(df, selected_dataset):
    """Create an enhanced timeline with confidence intervals and trend lines"""
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    colors = px.colors.qualitative.Set2
    
    fig = go.Figure()
    
    for i, metric in enumerate(metrics):
        # Add main metric line
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df[metric],
            name=metric.capitalize(),
            line=dict(color=colors[i], width=2),
            hovertemplate=f"{metric}: %{{y:.3f}}<br>Commit: %{{customdata}}<extra></extra>",
            customdata=df['commit_message']
        ))
        
        # Add trend line
        z = np.polyfit(range(len(df)), df[metric], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=p(range(len(df))),
            name=f'{metric.capitalize()} Trend',
            line=dict(color=colors[i], width=1, dash='dash'),
            showlegend=False
        ))
        
        # Add confidence interval
        confidence = 0.95
        std_err = stats.sem(df[metric])
        ci = std_err * stats.t.ppf((1 + confidence) / 2, len(df) - 1)
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'].tolist() + df['timestamp'].tolist()[::-1],
            y=(df[metric] + ci).tolist() + (df[metric] - ci).tolist()[::-1],
            fill='toself',
            fillcolor=colors[i],
            line=dict(color='rgba(255,255,255,0)'),
            name=f'{metric.capitalize()} CI',
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"Enhanced Metrics Timeline - {DATASET_DESCRIPTIONS[selected_dataset]['name']}",
        xaxis_title="Timestamp",
        yaxis_title="Score",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_commit_impact_analysis(df):
    """Create a visualization showing the impact of each commit on metrics"""
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    
    # Calculate metric changes between consecutive commits
    changes = df[metrics].diff()
    
    fig = go.Figure()
    
    for metric in metrics:
        fig.add_trace(go.Bar(
            x=df['commit_hash'].str[:7],
            y=changes[metric],
            name=metric.capitalize(),
            text=changes[metric].round(3),
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Commit Impact Analysis',
        xaxis_title='Commit Hash',
        yaxis_title='Metric Change',
        barmode='group',
        height=400
    )
    
    return fig

def main():
    # Main dashboard
    st.set_page_config(
        layout="wide",
        page_title="ML Experiment Metrics Dashboard",
        page_icon="📊"
    )
    
    st.title("ML Experiment Metrics Dashboard")
    
    # Model Overview Section
    st.header("Model Overview")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Architecture")
        st.markdown(MODEL_OVERVIEW["architecture"]["description"])
        st.subheader("Training Configuration")
        st.markdown(MODEL_OVERVIEW["architecture"]["training"])
    
    with col2:
        st.subheader("Performance Metrics")
        for metric, description in MODEL_OVERVIEW["metrics"].items():
            with st.expander(f"{metric.capitalize()}"):
                st.markdown(description)
    
    # Load data
    df = load_data(METRICS_FILE)
    
    # Check if we have any data
    if df.empty:
        st.warning("No metrics data available. Please ensure the metrics file is properly populated.")
        return
        
    # Add aggregate score
    df['aggregate_score'] = df.apply(
        lambda x: calculate_aggregate_score(x[['accuracy', 'f1', 'precision', 'recall']]),
        axis=1
    )
    
    # Dataset selector
    selected_dataset = st.selectbox(
        "Select Dataset",
        df['dataset'].unique(),
        format_func=lambda x: DATASET_DESCRIPTIONS[x]['name']
    )
    
    # Display dataset description
    st.markdown(f"""
    ### {DATASET_DESCRIPTIONS[selected_dataset]['name']}
    
    **Description:** {DATASET_DESCRIPTIONS[selected_dataset]['description']}
    
    **Use Case:** {DATASET_DESCRIPTIONS[selected_dataset]['use_case']}
    
    **Interpretation Guide:** {DATASET_DESCRIPTIONS[selected_dataset]['interpretation']}
    """)
    
    # Filter data for selected dataset
    dataset_df = df[df['dataset'] == selected_dataset]
    
    # Create two columns for the layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Enhanced timeline with confidence intervals and trend lines
        fig = create_enhanced_timeline(dataset_df, selected_dataset)
        st.plotly_chart(fig, use_container_width=True)
        
        # Network graph showing commit relationships
        network_fig = create_network_graph(dataset_df, selected_dataset)
        st.plotly_chart(network_fig, use_container_width=True)
        
        # Commit impact analysis
        impact_fig = create_commit_impact_analysis(dataset_df)
        st.plotly_chart(impact_fig, use_container_width=True)
    
    with col2:
        # Aggregate score card
        latest_score = dataset_df['aggregate_score'].iloc[-1]
        score_change = latest_score - dataset_df['aggregate_score'].iloc[-2]
        
        st.metric(
            "Aggregate Performance Score",
            f"{latest_score:.3f}",
            f"{score_change:+.3f}",
            delta_color="normal"
        )
        
        # Latest metrics table
        latest_metrics = dataset_df[['accuracy', 'f1', 'precision', 'recall']].iloc[-1]
        st.table(latest_metrics.round(3))
        
        # Radar chart for comprehensive metric comparison
        radar_fig = create_radar_chart(dataset_df, selected_dataset)
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Metric correlation heatmap
        heatmap_fig = create_metric_correlation_heatmap(dataset_df)
        st.plotly_chart(heatmap_fig, use_container_width=True)
    
    # Commit history and changes
    st.subheader("Commit History and Significant Changes")
    
    for i in range(len(dataset_df)-1, 0, -1):
        current = dataset_df.iloc[i]
        previous = dataset_df.iloc[i-1]
        
        # Calculate metric changes
        changes = {metric: current[metric] - previous[metric] 
                  for metric in ['accuracy', 'f1', 'precision', 'recall']}
        
        # Only show commits with significant changes
        if any(abs(change) > 0.01 for change in changes.values()):
            commit_url = get_github_commit_url(current['commit_hash'])
            with st.expander(f"📝 [{current['commit_hash'][:7]}]({commit_url}) - {current['commit_message']}"):
                cols = st.columns(len(changes))
                for col, (metric, change) in zip(cols, changes.items()):
                    col.metric(
                        metric.capitalize(),
                        f"{current[metric]:.3f}",
                        f"{change:+.3f}",
                        delta_color="normal"
                    )
    
    # Cross-dataset comparison
    st.subheader("Cross-Dataset Performance Comparison")
    latest_data = df.groupby('dataset').last().reset_index()
    
    fig = go.Figure(data=[
        go.Bar(
            name=metric.capitalize(),
            x=[DATASET_DESCRIPTIONS[d]['name'] for d in latest_data['dataset']],
            y=latest_data[metric],
            text=latest_data[metric].round(3),
            textposition='auto',
        ) for metric in ['accuracy', 'f1', 'precision', 'recall']
    ])
    
    fig.update_layout(
        barmode='group',
        title="Latest Metrics Across All Datasets",
        xaxis_title="Dataset",
        yaxis_title="Score",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 