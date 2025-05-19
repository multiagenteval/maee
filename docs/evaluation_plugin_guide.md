# Creating Evaluation Plugins for MAEE

This guide explains how to create new evaluation plugins for the MAEE system.

## Overview

Evaluation plugins in MAEE are Python classes that implement specific evaluation logic. Each plugin is associated with a YAML configuration file that defines its metadata, required context, and metrics.

## Plugin Structure

A typical evaluation plugin has the following structure:

```python
from maee_agent.core.plugin_base import EvaluationPlugin
from maee_agent.core.metrics import MetricsCollector
from maee_agent.core.tracing import TracingProvider

class MyEvaluationPlugin(EvaluationPlugin):
    def __init__(self, metrics_collector: MetricsCollector, tracing_provider: TracingProvider):
        super().__init__(metrics_collector, tracing_provider)
        
    def evaluate(self, context: dict) -> dict:
        """
        Perform the evaluation using the provided context.
        
        Args:
            context: Dictionary containing required context fields
            
        Returns:
            Dictionary containing evaluation results and metrics
        """
        with self.tracing_provider.start_span("my_evaluation"):
            # Perform evaluation logic here
            result = self._run_evaluation(context)
            
            # Record metrics
            self.metrics_collector.record_metric(
                name="my_metric",
                value=result["value"],
                tags={"context_field": context["field"]}
            )
            
            return result
            
    def _run_evaluation(self, context: dict) -> dict:
        """Internal method to run the evaluation logic"""
        # Implementation here
        pass
```

## Creating a New Plugin

1. Use the CLI to generate a template:
   ```bash
   python -m maee_agent.cli create my_evaluation --category accuracy
   ```

2. This will create two files:
   - `maee_agent/evaluations/my_evaluation.yaml`: Configuration file
   - `maee_agent/plugins/my_evaluation.py`: Plugin implementation

3. Edit the YAML file to define your evaluation:
   ```yaml
   name: my_evaluation
   category: accuracy
   description: My custom evaluation
   priority: 1
   implementation:
     plugin_file: maee_agent/plugins/my_evaluation.py
     plugin_class: MyEvaluationPlugin
   required_context:
     - field1
     - field2
   metrics:
     - name: my_metric
       description: Description of my metric
       unit: percentage
       tags:
         - tag1
         - tag2
       thresholds:
         warning: 0.95
         critical: 0.90
   prompt_template: |
     Analyze the code changes for potential impact on my evaluation.
     Consider:
     1. First consideration
     2. Second consideration
     3. Third consideration

     Required context: {required_context}
     Available metrics: {metrics}

     Based on the changes, determine if my evaluation is necessary.
   ```

4. Implement the plugin class in the Python file:
   - Inherit from `EvaluationPlugin`
   - Implement the `evaluate` method
   - Use `metrics_collector` to record metrics
   - Use `tracing_provider` for tracing

## Best Practices

1. **Error Handling**:
   ```python
   def evaluate(self, context: dict) -> dict:
       try:
           # Validation
           self._validate_context(context)
           
           # Evaluation logic
           result = self._run_evaluation(context)
           
           return result
       except Exception as e:
           self.metrics_collector.record_metric(
               name="evaluation_error",
               value=1,
               tags={"error": str(e)}
           )
           raise
   ```

2. **Context Validation**:
   ```python
   def _validate_context(self, context: dict):
       required_fields = ["field1", "field2"]
       missing_fields = [field for field in required_fields if field not in context]
       if missing_fields:
           raise ValueError(f"Missing required fields: {missing_fields}")
   ```

3. **Metric Recording**:
   ```python
   def _record_metrics(self, result: dict, context: dict):
       # Record main metric
       self.metrics_collector.record_metric(
           name="my_metric",
           value=result["value"],
           tags={"context_field": context["field"]}
       )
       
       # Record additional metrics
       for key, value in result.get("additional_metrics", {}).items():
           self.metrics_collector.record_metric(
               name=key,
               value=value,
               tags={"context_field": context["field"]}
           )
   ```

4. **Tracing**:
   ```python
   def evaluate(self, context: dict) -> dict:
       with self.tracing_provider.start_span("my_evaluation"):
           # Record span attributes
           self.tracing_provider.set_attribute("context_field", context["field"])
           
           # Evaluation logic
           result = self._run_evaluation(context)
           
           # Record result attributes
           self.tracing_provider.set_attribute("result_value", result["value"])
           
           return result
   ```

## Testing Your Plugin

1. Create a test file:
   ```python
   import pytest
   from maee_agent.plugins.my_evaluation import MyEvaluationPlugin
   from maee_agent.core.metrics import MetricsCollector
   from maee_agent.core.tracing import TracingProvider

   @pytest.fixture
   def plugin():
       metrics_collector = MetricsCollector()
       tracing_provider = TracingProvider()
       return MyEvaluationPlugin(metrics_collector, tracing_provider)

   def test_evaluate(plugin):
       context = {
           "field1": "value1",
           "field2": "value2"
       }
       
       result = plugin.evaluate(context)
       
       assert "value" in result
       assert result["value"] >= 0
       assert result["value"] <= 1
   ```

2. Run the tests:
   ```bash
   pytest tests/test_my_evaluation.py
   ```

## Validation

Before using your plugin:

1. Validate the YAML configuration:
   ```bash
   python -m maee_agent.cli validate
   ```

2. Check the implementation:
   ```bash
   python -m maee_agent.cli check
   ```

## Integration

Your plugin will be automatically discovered and loaded by the evaluation registry. The LLM will use your plugin's prompt template to determine when to run your evaluation.

## Example Plugins

See the following example plugins for reference:
- `maee_agent/plugins/accuracy_evaluation.py`
- `maee_agent/plugins/performance_evaluation.py` 