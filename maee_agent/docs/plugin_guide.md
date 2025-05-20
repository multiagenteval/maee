# MAEE Evaluation Plugin Guide

This guide explains how to create and integrate evaluation plugins for the MAEE (Model Analysis and Evaluation Ecosystem) framework.

## Overview

MAEE uses a plugin-based architecture for evaluations, allowing you to easily add new types of evaluations without modifying the core framework. Each plugin is responsible for:

1. Defining its required context and dependencies
2. Validating input parameters
3. Running the evaluation
4. Reporting metrics and traces
5. Returning results in a standardized format

## Creating a New Plugin

### Basic Structure

Here's a template for creating a new evaluation plugin:

```python
from typing import Dict, Any
from maee_agent.core.interfaces import EvaluationPlugin
from maee_agent.metrics.prometheus_collector import metrics_collector
from maee_agent.tracing.opentelemetry_provider import tracing_provider

class MyEvaluationPlugin(EvaluationPlugin):
    def __init__(self):
        self.required_context_fields = [
            "field1",
            "field2",
            # Add your required fields here
        ]
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": "my_evaluation",
            "version": "1.0.0",
            "description": "Description of what this evaluation does",
            "required_context": self.required_context_fields,
            "output_metrics": ["metric1", "metric2"]
        }
    
    def validate_context(self, context: Dict[str, Any]) -> bool:
        return all(field in context for field in self.required_context_fields)
    
    def run_evaluation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.validate_context(context):
            raise ValueError("Missing required context fields")
        
        # Start tracing
        span_id = tracing_provider.start_span("my_evaluation")
        try:
            # Record start time for metrics
            start_time = time.time()
            
            # Your evaluation logic here
            
            # Record metrics
            duration_ms = (time.time() - start_time) * 1000
            metrics_collector.record_timing(
                "my_evaluation_duration",
                duration_ms,
                {"context": "value"}
            )
            
            # Record other metrics
            metrics_collector.record_metric(
                "my_evaluation_metric",
                value,
                {"context": "value"}
            )
            
            # Add span attributes
            tracing_provider.add_span_attribute(span_id, "key", "value")
            
            return results
            
        except Exception as e:
            tracing_provider.end_span(span_id, status="error", error=e)
            raise
        finally:
            tracing_provider.end_span(span_id)
```

### Key Components

1. **Required Context Fields**
   - Define all required fields in `__init__`
   - Use clear, descriptive names
   - Document any special requirements or formats

2. **Metadata**
   - Provide clear name and version
   - List all required context fields
   - Document output metrics
   - Include any dependencies or prerequisites

3. **Context Validation**
   - Validate all required fields
   - Check data types and formats
   - Verify any constraints or requirements

4. **Evaluation Logic**
   - Implement your evaluation algorithm
   - Handle errors gracefully
   - Use appropriate logging

5. **Metrics and Tracing**
   - Record timing information
   - Track relevant metrics
   - Add meaningful span attributes
   - Handle errors properly

## Best Practices

1. **Error Handling**
   - Use specific exception types
   - Provide clear error messages
   - Clean up resources properly

2. **Performance**
   - Record timing metrics
   - Use appropriate batch sizes
   - Implement caching when possible

3. **Documentation**
   - Document all parameters
   - Explain the evaluation logic
   - Provide usage examples

4. **Testing**
   - Write unit tests
   - Include integration tests
   - Test error cases

## Registering Your Plugin

To register your plugin, add it to the plugin registry:

```python
from maee_agent.core.plugin_registry import plugin_registry
from my_plugin import MyEvaluationPlugin

plugin_registry.register_plugin("my_evaluation", MyEvaluationPlugin)
```

## Example: Accuracy Evaluation Plugin

See the `accuracy_evaluation.py` plugin for a complete example of a well-structured evaluation plugin.

## Metrics and Tracing

### Available Metrics

1. **Timing Metrics**
   ```python
   metrics_collector.record_timing("evaluation_duration", duration_ms)
   ```

2. **Value Metrics**
   ```python
   metrics_collector.record_metric("evaluation_score", value)
   ```

3. **Counter Metrics**
   ```python
   metrics_collector.record_counter("evaluation_count")
   ```

### Tracing

1. **Starting a Span**
   ```python
   span_id = tracing_provider.start_span("evaluation_name")
   ```

2. **Adding Attributes**
   ```python
   tracing_provider.add_span_attribute(span_id, "key", "value")
   ```

3. **Ending a Span**
   ```python
   tracing_provider.end_span(span_id)
   ```

## Testing Your Plugin

1. **Unit Tests**
   ```python
   def test_my_evaluation():
       plugin = MyEvaluationPlugin()
       context = {
           "field1": "value1",
           "field2": "value2"
       }
       results = plugin.run_evaluation(context)
       assert results["metric1"] > 0
   ```

2. **Integration Tests**
   ```python
   def test_plugin_integration():
       plugin = plugin_registry.create_plugin_instance("my_evaluation")
       results = plugin.run_evaluation(test_context)
       assert results is not None
   ```

## Common Issues and Solutions

1. **Missing Context Fields**
   - Always validate context before running
   - Provide clear error messages
   - Document required fields

2. **Performance Issues**
   - Use appropriate batch sizes
   - Implement caching
   - Profile your code

3. **Error Handling**
   - Use try/except blocks
   - Clean up resources
   - Report errors properly

## Support

For additional support or questions:
1. Check the documentation
2. Review example plugins
3. Contact the MAEE team 