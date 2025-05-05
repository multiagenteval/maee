import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import git
from openai import OpenAI
from dotenv import load_dotenv

# Get the root directory and load environment variables from root .env
ROOT_DIR = Path(__file__).parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

class MAEEAgent:
    def __init__(self, repo_path: str, model: str = "o3-mini"):
        """Initialize the MAEE agent with repository path and OpenAI model."""
        self.repo_path = Path(repo_path)
        self.model = model
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            organization="org-J0gyKgcFcDHR1iv8zJeyYBvl"
        )
        self.repo = git.Repo(repo_path)
        self.output_dir = self.repo_path / "output" / "workflow"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _log_openai_interaction(self, prompt: str, response: str, step: str):
        """Log OpenAI prompt and response to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"openai_logs_{timestamp}.json"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "model": self.model,
            "prompt": prompt,
            "response": response
        }
        
        with open(log_file, "w") as f:
            json.dump(log_entry, f, indent=2)
            
    def detect_changes(self) -> Dict[str, Any]:
        """Detect actual code changes in the repository."""
        # Get the current and previous commit
        current = self.repo.head.commit
        previous = current.parents[0] if current.parents else None
        
        if not previous:
            return {"changes_detected": False, "reason": "No previous commit found"}
            
        # Get the diff between commits
        diff = previous.diff(current)
        
        changes = {
            "changes_detected": len(diff) > 0,
            "files_changed": [],
            "diffs": {}
        }
        
        for d in diff:
            if d.a_path:
                changes["files_changed"].append(d.a_path)
                # Get the actual diff content
                changes["diffs"][d.a_path] = {
                    "old_content": d.a_blob.data_stream.read().decode('utf-8') if d.a_blob else "",
                    "new_content": d.b_blob.data_stream.read().decode('utf-8') if d.b_blob else "",
                    "change_type": d.change_type
                }
                
        return changes
        
    def analyze_changes(self, changes: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code changes using OpenAI API."""
        if not changes["changes_detected"]:
            return {"analysis": "No changes to analyze"}
            
        # Get available evaluations from config
        from ..config import EVAL_CATEGORIES
        required_evals = list(EVAL_CATEGORIES["required"].keys())
        optional_evals = list(EVAL_CATEGORIES["optional"].keys())
            
        # Prepare the prompt for OpenAI
        prompt = f"""Analyze the following code changes and determine which evaluations to run.

Available Evaluations:
Required:
{chr(10).join(f"- {eval_name}: {desc}" for eval_name, desc in EVAL_CATEGORIES["required"].items())}

Optional:
{chr(10).join(f"- {eval_name}: {desc}" for eval_name, desc in EVAL_CATEGORIES["optional"].items())}

Changes detected in files: {changes['files_changed']}

Diffs:
{json.dumps(changes['diffs'], indent=2)}

For each evaluation, determine if it should be run based on the changes.
For required evaluations, indicate if they are relevant to the changes.
For optional evaluations, recommend which ones to run based on the changes.

Provide your response in the following format:
1. Required Evaluations:
   - [eval_name]: [yes/no] - [reason]
2. Optional Evaluations:
   - [eval_name]: [yes/no] - [reason]
3. Additional Notes:
   - [any additional context or recommendations]
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI expert in code analysis and ML model evaluation."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Log the interaction
        self._log_openai_interaction(prompt, response.choices[0].message.content, "change_analysis")
        
        # Parse the response
        analysis = response.choices[0].message.content
        
        return {
            "raw_analysis": analysis,
            "structured_analysis": self._structure_analysis(analysis)
        }
        
    def plan_evaluations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan evaluations based on the analysis."""
        # Get available evaluations from config
        from ..config import EVAL_CATEGORIES
        
        # Parse the analysis to determine which evaluations to run
        required_evals = []
        optional_evals = []
        
        # Split the analysis into sections
        sections = analysis["raw_analysis"].split("\n\n")
        current_section = None
        
        for line in sections[0].split("\n"):
            if line.startswith("1. Required Evaluations:"):
                current_section = "required"
            elif line.startswith("2. Optional Evaluations:"):
                current_section = "optional"
            elif line.startswith("3. Additional Notes:"):
                break
            elif line.strip().startswith("- ") and current_section:
                eval_name = line.split(":")[0].strip("- ").strip()
                should_run = "yes" in line.lower()
                if should_run:
                    if current_section == "required":
                        required_evals.append(eval_name)
                    else:
                        optional_evals.append(eval_name)
        
        # Create evaluation plan
        plan = {
            "required_evaluations": required_evals,
            "optional_evaluations": optional_evals,
            "metrics": [],
            "test_datasets": [],
            "baseline_comparisons": [],
            "success_criteria": {}
        }
        
        # Add metrics and datasets based on selected evaluations
        for eval_name in required_evals + optional_evals:
            if eval_name in EVAL_CATEGORIES["required"]:
                eval_config = EVAL_CATEGORIES["required"][eval_name]
            else:
                eval_config = EVAL_CATEGORIES["optional"][eval_name]
            
            # Add metrics and datasets based on evaluation type
            if eval_name == "accuracy":
                plan["metrics"].extend(["accuracy", "precision", "recall", "f1"])
                plan["test_datasets"].append("test")
            elif eval_name == "performance":
                plan["metrics"].extend(["inference_time", "memory_usage", "throughput"])
                plan["test_datasets"].append("test")
            elif eval_name == "robustness":
                plan["metrics"].extend(["adversarial_accuracy", "noise_robustness"])
                plan["test_datasets"].append("test_adversarial")
            elif eval_name == "fairness":
                plan["metrics"].extend(["demographic_parity", "equal_opportunity"])
                plan["test_datasets"].append("test_balanced")
            elif eval_name == "interpretability":
                plan["metrics"].extend(["feature_importance", "decision_boundary"])
                plan["test_datasets"].append("test")
            elif eval_name == "safety":
                plan["metrics"].extend(["harmful_outputs", "bias_amplification"])
                plan["test_datasets"].append("test")
        
        # Add baseline comparisons and success criteria
        plan["baseline_comparisons"] = ["previous_commit", "production"]
        plan["success_criteria"] = {
            "accuracy": {"threshold": 0.95},
            "inference_time": {"threshold": 0.1},  # seconds
            "memory_usage": {"threshold": 1000}  # MB
        }
        
        return {
            "raw_plan": json.dumps(plan, indent=2),
            "structured_plan": plan
        }
        
    def execute_evaluations(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the evaluation plan and collect results."""
        import sys
        from pathlib import Path
        
        # Add demo_repo to Python path
        demo_repo_path = Path(self.repo_path)
        if str(demo_repo_path) not in sys.path:
            sys.path.append(str(demo_repo_path))
        
        from eval.experiment_tracker import ExperimentTracker
        
        # Initialize experiment tracker
        tracker = ExperimentTracker()
        
        # Execute evaluations and collect metrics
        metrics_by_dataset = {
            "test": {
                "accuracy": 0.75,  # Simulated regression from 0.95
                "precision": {
                    "class_0": 0.76,
                    "class_1": 0.74,
                    "class_2": 0.75
                },
                "training_time": 3.5,  # seconds per epoch
                "inference_time": 0.08,  # seconds per batch
                "memory_usage": 1.5  # GB
            }
        }
        
        # Save experiment results
        experiment = tracker.save_experiment(metrics_by_dataset, plan["structured_plan"], status="pending")
        
        # Compare with baseline
        comparisons = tracker.compare_with_baseline(metrics_by_dataset["test"])
        
        return {
            "status": "completed",
            "results": {
                "metrics": metrics_by_dataset,
                "comparisons": comparisons,
                "experiment": experiment
            }
        }
        
    def analyze_results(self, results: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze evaluation results and detect regressions."""
        # Extract metrics and comparisons
        metrics = results["results"]["metrics"]
        comparisons = results["results"]["comparisons"]
        
        # Create structured analysis
        structured_analysis = {
            "performance_changes": metrics,
            "regressions_detected": comparisons.get("overall_status") == "regression",
            "recommendations": []
        }
        
        # Add recommendations based on comparison status
        if comparisons.get("status") == "no_baseline":
            structured_analysis["recommendations"].extend([
                "Establish this as the baseline for future comparisons",
                "Monitor these metrics for future changes",
                "Set thresholds for acceptable performance"
            ])
        else:
            if comparisons.get("overall_status") == "regression":
                structured_analysis["recommendations"].extend([
                    "Monitor the identified regressions",
                    "Consider rolling back changes if regressions are significant",
                    "Investigate root causes of performance changes"
                ])
            else:
                structured_analysis["recommendations"].extend([
                    "Continue monitoring performance metrics",
                    "Document successful improvements",
                    "Consider setting new baseline thresholds"
                ])
        
        return {
            "raw_analysis": None,
            "structured_analysis": structured_analysis
        }
        
    def _structure_analysis(self, raw_analysis: str) -> Dict[str, Any]:
        """Structure the raw analysis text into a structured format."""
        # Use OpenAI to structure the raw text
        prompt = f"""Convert this analysis into a structured JSON format:

{raw_analysis}

Include:
- impact_areas: list
- risk_level: string
- required_evaluations: list
- potential_risks: list
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that converts text to structured JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Log the interaction
        self._log_openai_interaction(prompt, response.choices[0].message.content, "structure_analysis")
        
        return json.loads(response.choices[0].message.content)
        
    def _structure_evaluation_plan(self, raw_plan: str) -> Dict[str, Any]:
        """Structure the raw evaluation plan into a structured format."""
        prompt = f"""Convert this evaluation plan into a structured JSON format:

{raw_plan}

Include:
- metrics: list
- test_datasets: list
- baseline_comparisons: list
- success_criteria: object
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that converts text to structured JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Log the interaction
        self._log_openai_interaction(prompt, response.choices[0].message.content, "structure_evaluation_plan")
        
        return json.loads(response.choices[0].message.content)
        
    def _structure_results_analysis(self, raw_analysis: str) -> Dict[str, Any]:
        """Structure the raw results analysis into a structured format."""
        prompt = f"""Convert this results analysis into a structured JSON format:

{raw_analysis}

Include:
- performance_changes: object
- regressions_detected: boolean
- recommendations: list
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI assistant that converts text to structured JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Log the interaction
        self._log_openai_interaction(prompt, response.choices[0].message.content, "structure_results_analysis")
        
        return json.loads(response.choices[0].message.content)
        
    def run_workflow(self) -> Dict[str, Any]:
        """Run the complete workflow."""
        workflow_results = {
            "timestamp": datetime.now().isoformat(),
            "commit": self.repo.head.commit.hexsha,
            "steps": []
        }
        
        # 1. Detect Changes
        changes = self.detect_changes()
        workflow_results["steps"].append({"step": "change_detection", "results": changes})
        
        if not changes["changes_detected"]:
            return workflow_results
            
        # 2. Analyze Changes
        analysis = self.analyze_changes(changes)
        workflow_results["steps"].append({"step": "change_analysis", "results": analysis})
        
        # 3. Plan Evaluations
        plan = self.plan_evaluations(analysis)
        workflow_results["steps"].append({"step": "evaluation_planning", "results": plan})
        
        # 4. Execute Evaluations
        results = self.execute_evaluations(plan)
        workflow_results["steps"].append({"step": "evaluation_execution", "results": results})
        
        # 5. Analyze Results
        final_analysis = self.analyze_results(results)
        workflow_results["steps"].append({"step": "results_analysis", "results": final_analysis})
        
        # Save workflow results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"workflow_results_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(workflow_results, f, indent=2)
            
        return workflow_results 