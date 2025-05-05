import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import git
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MAEEAgent:
    def __init__(self, repo_path: str, model: str = "gpt-4-turbo-preview"):
        """Initialize the MAEE agent with repository path and OpenAI model."""
        self.repo_path = Path(repo_path)
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.repo = git.Repo(repo_path)
        self.output_dir = self.repo_path / "output" / "workflow"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
            
        # Prepare the prompt for OpenAI
        prompt = f"""Analyze the following code changes and provide:
1. Impact areas (e.g., model architecture, training, evaluation)
2. Risk level (low, medium, high)
3. Required evaluations
4. Potential risks

Changes detected in files: {changes['files_changed']}

Diffs:
{json.dumps(changes['diffs'], indent=2)}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI expert in code analysis and ML model evaluation."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        # Parse the response
        analysis = response.choices[0].message.content
        
        return {
            "raw_analysis": analysis,
            "structured_analysis": self._structure_analysis(analysis)
        }
        
    def plan_evaluations(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Plan evaluations based on the analysis."""
        prompt = f"""Based on the following analysis, create a detailed evaluation plan:

{json.dumps(analysis, indent=2)}

Include:
1. Required metrics to evaluate
2. Test datasets to use
3. Baseline comparisons needed
4. Success criteria
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI expert in ML model evaluation planning."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        plan = response.choices[0].message.content
        
        return {
            "raw_plan": plan,
            "structured_plan": self._structure_evaluation_plan(plan)
        }
        
    def execute_evaluations(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the evaluation plan and collect results."""
        # This would integrate with your actual model evaluation code
        # For now, we'll return a placeholder that you can extend
        return {
            "status": "not_implemented",
            "message": "Implement this method to run actual model evaluations based on the plan"
        }
        
    def analyze_results(self, results: Dict[str, Any], baseline: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Analyze evaluation results and detect regressions."""
        prompt = f"""Analyze these evaluation results and detect any regressions:

Current Results:
{json.dumps(results, indent=2)}

Baseline Results:
{json.dumps(baseline, indent=2) if baseline else "No baseline available"}

Provide:
1. Performance changes
2. Regression detection
3. Recommendations
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an AI expert in ML model evaluation analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        analysis = response.choices[0].message.content
        
        return {
            "raw_analysis": analysis,
            "structured_analysis": self._structure_results_analysis(analysis)
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
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
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
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
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
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
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