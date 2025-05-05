import graphviz
from pathlib import Path
import json
from typing import Dict, Any, List
from datetime import datetime
from enum import Enum
import statistics
import torch
import torch.optim as optim
import torch.nn as nn
import random
import string

from maee_agent.state_management import WorkflowStage

class AgentType(Enum):
    CODE_CHANGE_DETECTOR = "code_change_detector"
    DIFF_ANALYZER = "diff_analyzer"
    IMPACT_ANALYZER = "impact_analyzer"
    EVAL_PLANNER = "eval_planner"
    EVAL_EXECUTOR = "eval_executor"
    RESULTS_ANALYZER = "results_analyzer"
    REGRESSION_ANALYZER = "regression_analyzer"

class ResultsTracker:
    def __init__(self, results_file: Path):
        self.results_file = results_file
        self.results_file.parent.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        self.load_history()
    
    def load_history(self) -> None:
        """Load existing evaluation results history."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                self.history = json.load(f)
    
    def save_history(self) -> None:
        """Save evaluation results history."""
        with open(self.results_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def add_result(self, result: Dict[str, Any]) -> None:
        """Add a new evaluation result to the history."""
        result['timestamp'] = datetime.now().isoformat()
        self.history.append(result)
        self.save_history()
    
    def get_metric_history(self, metric_name: str) -> List[Dict[str, Any]]:
        """Get historical values for a specific metric with commit information."""
        return [
            {
                'value': eval_result['results'][metric_name],
                'commit_hash': eval_result['commit_hash'],
                'timestamp': eval_result['timestamp']
            }
            for eval_result in self.history
            if metric_name in eval_result['results']
        ]
    
    def detect_regression(self, metric_name: str, current_value: float, threshold: float = 0.05) -> Dict[str, Any]:
        """Detect if there's a regression in a metric."""
        history = self.get_metric_history(metric_name)
        if not history:
            return {"is_regression": False, "message": "No historical data available"}
        
        # Get the previous value (most recent before current)
        previous_value = history[-1]['value']
        
        # Calculate the percentage change from the previous value
        change = (current_value - previous_value) / previous_value if previous_value != 0 else 0
        
        is_regression = change < -threshold
        return {
            "is_regression": is_regression,
            "metric": metric_name,
            "current_value": current_value,
            "previous_value": previous_value,
            "change_percentage": change * 100,
            "threshold": threshold * 100,
            "previous_commit": history[-1]['commit_hash'],
            "message": f"Regression detected: {metric_name} decreased by {abs(change*100):.2f}% from {previous_value:.4f} to {current_value:.4f}" if is_regression else f"No regression detected in {metric_name}"
        }

class AgentLog:
    def __init__(self):
        self.logs: List[Dict[str, Any]] = []
        self.current_state: Dict[str, Any] = {}
        self.results_tracker = ResultsTracker(Path("output/workflow/evaluation_history.json"))
    
    def log_activity(self, 
                    agent_type: AgentType,
                    input_data: Dict[str, Any],
                    output_data: Dict[str, Any],
                    next_step: str,
                    timestamp: datetime = None) -> None:
        """Log an agent's activity with detailed information."""
        if timestamp is None:
            timestamp = datetime.now()
            
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "agent": agent_type.value,
            "input": input_data,
            "output": output_data,
            "next_step": next_step,
            "state": self.current_state.copy()
        }
        self.logs.append(log_entry)
        
        # If this is an evaluation result, track it
        if agent_type == AgentType.EVAL_EXECUTOR:
            self.results_tracker.add_result({
                "commit_hash": self.current_state.get("commit_hash"),
                "evaluation_type": input_data.get("evaluation_plan"),
                "results": output_data.get("results", {})
            })
    
    def update_state(self, new_state: Dict[str, Any]) -> None:
        """Update the current state of the workflow."""
        self.current_state.update(new_state)
    
    def save_logs(self, output_dir: Path) -> None:
        """Save the logs to a JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        log_file = output_dir / f'agent_logs_{timestamp}.json'
        with open(log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        
        print(f"\nAgent logs have been saved to {log_file}")

def generate_commit_hash() -> str:
    """Generate a random commit hash for simulation purposes."""
    return ''.join(random.choices(string.hexdigits.lower(), k=40))

def simulate_agent_workflow(command_info=None):
    """Simulate the agent workflow and generate logs"""
    # Initialize logging
    agent_log = AgentLog()
    
    # First establish baseline with good performance
    update_evaluation_history(
        commit_hash="ec59f1a2db90eba3b0a04bbdc16edc20e5d17db2",
        evaluation_type="accuracy",
        results={
            "overall_accuracy": 0.95,
            "class_wise_precision": {
                "class_0": 0.96,
                "class_1": 0.94,
                "class_2": 0.95
            }
        }
    )
    
    # Simulate code change detection
    agent_log.log_activity(
        AgentType.CODE_CHANGE_DETECTOR,
        {
            "repository_path": "demo_repo", 
            "branch": "main",
            "command_info": command_info
        },
        {
            "changes_detected": True,
            "files_changed": ["train.py"],
            "change_type": "model_metrics",
            "diff": {
                "train.py": {
                    "old_content": "def train_model(model, train_loader, val_loader, num_epochs=10):\n    criterion = nn.CrossEntropyLoss()\n    optimizer = optim.Adam(model.parameters(), lr=0.01)",
                    "new_content": "def train_model(model, train_loader, val_loader, num_epochs=10):\n    criterion = nn.CrossEntropyLoss()\n    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimized learning rate\n    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)  # Added learning rate scheduling",
                    "changes": [
                        {
                            "type": "modified",
                            "line": 2,
                            "old_content": "optimizer = optim.Adam(model.parameters(), lr=0.01)",
                            "new_content": "optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimized learning rate"
                        },
                        {
                            "type": "added",
                            "line": 3,
                            "content": "scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)  # Added learning rate scheduling"
                        }
                    ]
                }
            }
        },
        "diff_analysis"
    )
    
    # Simulate diff analysis
    agent_log.log_activity(
        AgentType.DIFF_ANALYZER,
        {
            "changes": {
                "train.py": {
                    "changes": [
                        {
                            "type": "modified",
                            "line": 2,
                            "old_content": "optimizer = optim.Adam(model.parameters(), lr=0.01)",
                            "new_content": "optimizer = optim.Adam(model.parameters(), lr=0.001)  # Optimized learning rate"
                        },
                        {
                            "type": "added",
                            "line": 3,
                            "content": "scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)  # Added learning rate scheduling"
                        }
                    ]
                }
            }
        },
        {
            "impact_areas": ["model_architecture", "training_parameters"],
            "risk_level": "low",
            "key_changes": [
                "Reduced learning rate for better convergence",
                "Added learning rate scheduling for adaptive optimization"
            ]
        },
        "impact_analysis"
    )
    
    # Simulate impact analysis
    agent_log.log_activity(
        AgentType.IMPACT_ANALYZER,
        {
            "changes": {
                "impact_areas": ["model_architecture", "training_parameters"],
                "risk_level": "low"
            }
        },
        {
            "required_evaluations": [
                {
                    "type": "accuracy",
                    "priority": "high",
                    "reason": "Model architecture changes"
                },
                {
                    "type": "performance",
                    "priority": "medium",
                    "reason": "Training parameter optimization"
                }
            ],
            "optional_evaluations": [
                {
                    "type": "robustness",
                    "priority": "low",
                    "reason": "Parameter tuning"
                }
            ]
        },
        "eval_planning"
    )
    
    # Simulate evaluation planning
    agent_log.log_activity(
        AgentType.EVAL_PLANNER,
        {
            "required_evaluations": [
                {
                    "type": "accuracy",
                    "priority": "high"
                },
                {
                    "type": "performance",
                    "priority": "medium"
                }
            ]
        },
        {
            "evaluation_plan": {
                "accuracy": {
                    "metrics": ["overall_accuracy", "class_wise_precision"],
                    "test_sets": ["validation", "test"],
                    "baseline_comparison": True
                },
                "performance": {
                    "metrics": ["training_time", "inference_time", "memory_usage"],
                    "test_sets": ["validation"],
                    "baseline_comparison": True
                }
            }
        },
        "eval_execution"
    )
    
    # Simulate evaluation execution
    agent_log.log_activity(
        AgentType.EVAL_EXECUTOR,
        {
            "evaluation_plan": {
                "accuracy": {
                    "metrics": ["overall_accuracy", "class_wise_precision"],
                    "test_sets": ["validation", "test"]
                }
            }
        },
        {
            "results": {
                "accuracy": {
                    "overall_accuracy": 0.98,  # Improved accuracy
                    "class_wise_precision": {
                        "class_0": 0.99,  # Improved precision
                        "class_1": 0.97,  # Improved precision
                        "class_2": 0.98   # Improved precision
                    }
                },
                "performance": {
                    "training_time": "2.5s/epoch",  # Faster training
                    "inference_time": "0.05s/batch",  # Faster inference
                    "memory_usage": "1.2GB"  # Optimized memory
                }
            },
            "status": "completed"
        },
        "results_analysis"
    )
    
    # Simulate results analysis
    agent_log.log_activity(
        AgentType.RESULTS_ANALYZER,
        {
            "results": {
                "accuracy": {
                    "overall_accuracy": 0.98,
                    "class_wise_precision": {
                        "class_0": 0.99,
                        "class_1": 0.97,
                        "class_2": 0.98
                    }
                },
                "performance": {
                    "training_time": "2.5s/epoch",
                    "inference_time": "0.05s/batch",
                    "memory_usage": "1.2GB"
                }
            }
        },
        {
            "summary": "Significant performance improvement detected",
            "regression_analysis": {
                "overall_accuracy": {
                    "current": 0.98,
                    "previous": 0.95,
                    "change": "+0.03",
                    "trend": "improvement"
                },
                "class_wise_precision": {
                    "class_0": {"current": 0.99, "previous": 0.96, "change": "+0.03"},
                    "class_1": {"current": 0.97, "previous": 0.94, "change": "+0.03"},
                    "class_2": {"current": 0.98, "previous": 0.95, "change": "+0.03"}
                }
            },
            "recommendations": [
                "Keep the optimized learning rate and scheduler",
                "Consider applying similar optimizations to other models",
                "Document the performance improvements",
                "Monitor for any potential overfitting",
                "Consider running additional validation tests"
            ]
        },
        "complete"
    )
    
    # Save logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"output/workflow/agent_logs_{timestamp}.json"
    agent_log.save_logs(Path("output/workflow"))
    print(f"\nAgent logs saved to: {log_file}")
    
    # Print regression analysis
    print("\nRegression Analysis:")
    print("Summary: Significant performance improvement detected")
    print("\nRecommendations:")
    for rec in [
        "Keep the optimized learning rate and scheduler",
        "Consider applying similar optimizations to other models",
        "Document the performance improvements",
        "Monitor for any potential overfitting",
        "Consider running additional validation tests"
    ]:
        print(f"- {rec}")

def visualize_workflow(workflow: Any, current_state: Dict[str, Any] = None) -> None:
    """Generate a visual representation of the complete agent workflow with all potential evaluations."""
    dot = graphviz.Digraph(comment='MAEE Agent Workflow')
    dot.attr(rankdir='TB')  # Top to bottom layout
    dot.attr('graph', fontsize='12', fontname='Arial')
    dot.attr('node', fontsize='10', fontname='Arial')
    
    # Main Process Flow
    with dot.subgraph(name='cluster_process') as c:
        c.attr(label='MAEE Agent Process Flow', style='filled', fillcolor='lightgray')
        c.attr('node', style='filled', fillcolor='white')
        
        # Code Change Detection
        c.node('code_change', 'Code Change Detection', shape='box', style='filled', fillcolor='lightblue')
        
        # Analysis Phase
        c.node('diff_analysis', 'Diff Analysis\n(LLM-based)', shape='box', style='filled', fillcolor='lightgreen')
        c.node('impact_analysis', 'Impact Analysis\n(Code Changes)', shape='box', style='filled', fillcolor='lightgreen')
        
        # Evaluation Planning
        c.node('eval_planning', 'Evaluation Planning\n(Required vs Optional)', shape='diamond', style='filled', fillcolor='lightyellow')
        
        # Execution Phase
        c.node('eval_execution', 'Evaluation Execution', shape='box', style='filled', fillcolor='orange')
        c.node('results_analysis', 'Results Analysis', shape='box', style='filled', fillcolor='lightpink')
        
        # Process Flow Edges
        c.edge('code_change', 'diff_analysis', 'Code Changes Detected')
        c.edge('diff_analysis', 'impact_analysis', 'Change Analysis')
        c.edge('impact_analysis', 'eval_planning', 'Impact Assessment')
        c.edge('eval_planning', 'eval_execution', 'Plan Created')
        c.edge('eval_execution', 'results_analysis', 'Results Collected')
    
    # Required Evaluations
    with dot.subgraph(name='cluster_required') as c:
        c.attr(label='Required Evaluations', style='filled', fillcolor='lightpink')
        c.attr('node', style='filled', fillcolor='white')
        
        # Accuracy Evaluations
        c.node('acc_eval', 'Accuracy Evaluations', shape='box')
        c.node('acc_metrics', '• Model Accuracy\n• Per-class Accuracy\n• Precision/Recall\n• F1 Score', shape='note')
        
        # Performance Evaluations
        c.node('perf_eval', 'Performance Evaluations', shape='box')
        c.node('perf_metrics', '• Training Speed\n• Inference Time\n• Resource Usage\n• Throughput', shape='note')
        
        # Robustness Evaluations
        c.node('robust_eval', 'Robustness Evaluations', shape='box')
        c.node('robust_metrics', '• Error Handling\n• Edge Cases\n• Input Validation\n• Stability', shape='note')
        
        # Security Evaluations
        c.node('sec_eval', 'Security Evaluations', shape='box')
        c.node('sec_metrics', '• Vulnerability Scan\n• Access Control\n• Data Protection\n• API Security', shape='note')
        
        # Code Quality Evaluations
        c.node('code_eval', 'Code Quality Evaluations', shape='box')
        c.node('code_metrics', '• Code Style\n• Complexity\n• Test Coverage\n• Documentation', shape='note')
    
    # Optional Evaluations
    with dot.subgraph(name='cluster_optional') as c:
        c.attr(label='Optional Evaluations', style='filled', fillcolor='lightblue')
        c.attr('node', style='filled', fillcolor='white')
        
        # Model Complexity
        c.node('complex_eval', 'Model Complexity', shape='box')
        c.node('complex_metrics', '• Parameter Count\n• Layer Depth\n• Architecture\n• Memory Footprint', shape='note')
        
        # Scalability
        c.node('scale_eval', 'Scalability', shape='box')
        c.node('scale_metrics', '• Batch Processing\n• Distributed Training\n• Load Testing\n• Resource Scaling', shape='note')
        
        # Documentation
        c.node('doc_eval', 'Documentation', shape='box')
        c.node('doc_metrics', '• API Docs\n• Code Comments\n• Usage Examples\n• Tutorials', shape='note')
    
    # Connect Evaluations to Execution
    dot.edge('eval_planning', 'acc_eval', 'Required')
    dot.edge('eval_planning', 'perf_eval', 'Required')
    dot.edge('eval_planning', 'robust_eval', 'Required')
    dot.edge('eval_planning', 'sec_eval', 'Required')
    dot.edge('eval_planning', 'code_eval', 'Required')
    dot.edge('eval_planning', 'complex_eval', 'Optional')
    dot.edge('eval_planning', 'scale_eval', 'Optional')
    dot.edge('eval_planning', 'doc_eval', 'Optional')
    
    # Connect Evaluations to Execution
    dot.edge('acc_eval', 'eval_execution', 'Run')
    dot.edge('perf_eval', 'eval_execution', 'Run')
    dot.edge('robust_eval', 'eval_execution', 'Run')
    dot.edge('sec_eval', 'eval_execution', 'Run')
    dot.edge('code_eval', 'eval_execution', 'Run')
    dot.edge('complex_eval', 'eval_execution', 'Run')
    dot.edge('scale_eval', 'eval_execution', 'Run')
    dot.edge('doc_eval', 'eval_execution', 'Run')
    
    # Add Legend
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Legend', style='filled', fillcolor='lightgray')
        c.attr('node', style='filled', fillcolor='white')
        c.node('legend_process', 'Process Flow', shape='box', style='filled', fillcolor='lightblue')
        c.node('legend_required', 'Required Evaluations', shape='box', style='filled', fillcolor='lightpink')
        c.node('legend_optional', 'Optional Evaluations', shape='box', style='filled', fillcolor='lightblue')
        c.node('legend_metrics', 'Evaluation Metrics', shape='note')
    
    # Save the graph
    output_dir = Path("output/workflow")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dot.render(output_dir / f'workflow_graph_{timestamp}', format='png', cleanup=True)
    
    # Save current state alongside the graph
    if current_state:
        with open(output_dir / f'workflow_state_{timestamp}.json', 'w') as f:
            json.dump(current_state, f, indent=2)
    
    print(f"\nWorkflow visualization has been saved to {output_dir}/workflow_graph_{timestamp}.png")
    if current_state:
        print(f"Current state has been saved to {output_dir}/workflow_state_{timestamp}.json")
    
    # Also generate the agent logs
    simulate_agent_workflow()

class ResultsAnalyzer:
    def __init__(self):
        self.history_file = Path("output/workflow/evaluation_history.json")
        self.regression_threshold = 0.05  # 5% decrease in accuracy considered regression
        
    def analyze(self, results):
        # Load historical data
        historical_data = self._load_historical_data()
        
        # Get current metrics
        current_accuracy = results.get("accuracy", 0)
        current_class_precision = results.get("class_wise_precision", {})
        
        # Compare with historical data
        regression_analysis = self._detect_regression(historical_data, current_accuracy, current_class_precision)
        
        return {
            "summary": regression_analysis["summary"],
            "recommendations": regression_analysis["recommendations"],
            "regression_details": regression_analysis["details"]
        }
    
    def _load_historical_data(self):
        try:
            with open(self.history_file, 'r') as f:
                history = json.load(f)
                # Filter out entries with null commit hash
                return [entry for entry in history if entry["commit_hash"] is not None]
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def _detect_regression(self, historical_data, current_accuracy, current_class_precision):
        if not historical_data:
            return {
                "summary": "No historical data available for comparison",
                "recommendations": ["Establish baseline metrics"],
                "details": {"status": "no_history"}
            }
        
        # Get the baseline data (first evaluation)
        baseline = historical_data[0]
        baseline_accuracy = baseline["results"]["overall_accuracy"]
        baseline_class_precision = baseline["results"]["class_wise_precision"]
        
        # Calculate accuracy regression from baseline
        accuracy_decrease = baseline_accuracy - current_accuracy
        accuracy_regression = accuracy_decrease > self.regression_threshold
        
        # Calculate class-wise regression from baseline
        class_regressions = {}
        for class_name, current_precision in current_class_precision.items():
            baseline_precision = baseline_class_precision.get(class_name, 0)
            class_decrease = baseline_precision - current_precision
            class_regressions[class_name] = {
                "decrease": class_decrease,
                "is_regression": class_decrease > self.regression_threshold
            }
        
        # Determine regression severity
        if accuracy_regression or any(r["is_regression"] for r in class_regressions.values()):
            severity = "critical" if accuracy_decrease > 0.1 else "moderate"
            
            summary = f"{severity.title()} regression detected in model performance"
            recommendations = [
                "Revert recent model changes",
                "Review training parameters",
                "Check data quality and preprocessing"
            ]
            
            if severity == "critical":
                recommendations.extend([
                    "Immediate rollback recommended",
                    "Conduct root cause analysis",
                    "Run additional validation tests"
                ])
            
            return {
                "summary": summary,
                "recommendations": recommendations,
                "details": {
                    "status": "regression_detected",
                    "severity": severity,
                    "accuracy_decrease": accuracy_decrease,
                    "baseline_accuracy": baseline_accuracy,
                    "current_accuracy": current_accuracy,
                    "class_regressions": class_regressions,
                    "baseline_commit": baseline["commit_hash"]
                }
            }
        else:
            return {
                "summary": "No significant regression detected",
                "recommendations": ["Continue monitoring performance"],
                "details": {
                    "status": "stable",
                    "accuracy_decrease": accuracy_decrease,
                    "baseline_accuracy": baseline_accuracy,
                    "current_accuracy": current_accuracy,
                    "class_regressions": class_regressions,
                    "baseline_commit": baseline["commit_hash"]
                }
            }

def update_evaluation_history(commit_hash: str, evaluation_type: str, results: dict) -> None:
    """Update the evaluation history with new results"""
    history_file = Path("output/workflow/evaluation_history.json")
    history_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        history = []
    
    history.append({
        "commit_hash": commit_hash,
        "evaluation_type": evaluation_type,
        "results": results,
        "timestamp": datetime.now().isoformat()
    })
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2) 