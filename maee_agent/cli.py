import click
import json
import sys
from pathlib import Path
from datetime import datetime
from maee_agent.core.agent import MAEEAgent

@click.group()
def cli():
    """MAEE Agent CLI - Real implementation with OpenAI and Git integration"""
    pass

@cli.command()
@click.option('--evaluation-type', '-t', type=click.Choice(['accuracy', 'performance', 'robustness', 'all']), 
              default='all', help='Type of evaluation to run')
@click.option('--model', '-m', default='gpt-4-turbo-preview', 
              help='OpenAI model to use for analysis')
def run(evaluation_type, model):
    """Run the MAEE agent workflow with real OpenAI analysis"""
    try:
        # Initialize the agent with the current directory
        agent = MAEEAgent(repo_path=".", model=model)
        
        print(f"Running MAEE workflow on commit: {agent.repo.head.commit.hexsha}")
        print(f"Using OpenAI model: {model}")
        print("Running MAEE agent workflow...")
        
        # Run the workflow
        results = agent.run_workflow()
        
        # Print summary
        if results["steps"]:
            last_step = results["steps"][-1]
            if "results" in last_step and "raw_analysis" in last_step["results"]:
                print("\nAnalysis Summary:")
                print(last_step["results"]["raw_analysis"])
        
        print(f"\nWorkflow results saved to: {agent.output_dir}")
        
    except Exception as e:
        print(f"Error running workflow: {e}")
        raise

@cli.command()
@click.option('--results-file', '-f', type=click.Path(exists=True), 
              help='Path to specific results file to view')
@click.option('--summary', '-s', is_flag=True, 
              help='Show only the final analysis')
def view(results_file, summary):
    """View results from the agent workflow"""
    try:
        if results_file:
            results_path = Path(results_file)
        else:
            # Find the most recent results file
            results_dir = Path("output/workflow")
            results_files = list(results_dir.glob("workflow_results_*.json"))
            if not results_files:
                print("No results files found")
                return
            results_path = max(results_files, key=lambda x: x.stat().st_mtime)
        
        with open(results_path) as f:
            results = json.load(f)
            
        if summary:
            # Show only the final analysis
            if results["steps"]:
                last_step = results["steps"][-1]
                print(f"\nCommit: {results['commit']}")
                print(f"Timestamp: {results['timestamp']}")
                print("\nFinal Analysis:")
                if "results" in last_step and "raw_analysis" in last_step["results"]:
                    print(last_step["results"]["raw_analysis"])
        else:
            # Show complete workflow results
            print(f"\nWorkflow Results for commit {results['commit']}")
            print(f"Timestamp: {results['timestamp']}")
            
            for step in results["steps"]:
                print(f"\nStep: {step['step']}")
                print("Results:")
                print(json.dumps(step["results"], indent=2))
                
    except Exception as e:
        print(f"Error viewing results: {e}")
        raise

@cli.command()
@click.argument('commit_range', required=False)
def history(commit_range):
    """View evaluation history across commits"""
    try:
        history_file = Path("output/workflow/evaluation_history.json")
        if not history_file.exists():
            print("No evaluation history found")
            return
            
        with open(history_file) as f:
            history = json.load(f)
            
        print("\nEvaluation History:")
        for entry in history:
            print(f"\nCommit: {entry['commit_hash']}")
            print(f"Timestamp: {entry['timestamp']}")
            print("Results:")
            print(json.dumps(entry['results'], indent=2))
            
    except Exception as e:
        print(f"Error viewing history: {e}")
        raise

if __name__ == '__main__':
    cli() 