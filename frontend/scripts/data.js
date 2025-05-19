/**
 * Data handling for the MAEE Evaluation Dashboard
 */

class DataService {
    constructor() {
        this.workflowResults = [];
        this.selectedRunId = null;
        this.filterCriteria = {
            metrics: [],
            dateRange: '7d',
        };
    }

    /**
     * Load all workflow results from the API
     */
    async loadWorkflowResults() {
        try {
            // Fetch workflow results from the API
            const response = await fetch('/api/workflow-results');
            const data = await response.json();
            this.workflowResults = data.sort((a, b) => {
                return new Date(b.timestamp) - new Date(a.timestamp);
            });
            return this.workflowResults;
        } catch (error) {
            console.error('Error loading workflow results:', error);
            // For the demo, mock some data based on the JSON we've seen
            this.workflowResults = this.getMockWorkflowResults();
            return this.workflowResults;
        }
    }

    /**
     * Get a specific workflow result by its ID
     */
    getWorkflowResult(id) {
        return this.workflowResults.find(result => result.workflow_id === id);
    }

    /**
     * Set the currently selected run
     */
    selectRun(id) {
        this.selectedRunId = id;
        return this.getWorkflowResult(id);
    }

    /**
     * Get all unique metrics across all workflow results
     */
    getAllMetrics() {
        const requiredMetrics = new Set();
        const optionalMetrics = new Set();

        this.workflowResults.forEach(result => {
            // Add required metrics
            if (result.required_metrics) {
                Object.keys(result.required_metrics).forEach(metric => {
                    requiredMetrics.add(metric);
                });
            }
            
            // Add optional metrics
            if (result.optional_metrics) {
                Object.keys(result.optional_metrics).forEach(metric => {
                    optionalMetrics.add(metric);
                });
            }
        });

        return {
            required: Array.from(requiredMetrics),
            optional: Array.from(optionalMetrics)
        };
    }

    /**
     * Get aggregated metrics data for charting
     */
    getMetricsData(metricName) {
        const metricsData = [];
        
        this.workflowResults.forEach(result => {
            if (result.evaluation_results && result.evaluation_results[metricName]) {
                const timestamp = new Date(result.timestamp);
                
                // Find the main value for the metric
                let value = null;
                const metricResult = result.evaluation_results[metricName];
                
                // Different formats observed in the data
                if (metricResult.value !== undefined) {
                    value = metricResult.value;
                } else if (metricResult.metrics && metricResult.metrics.accuracy !== undefined) {
                    value = metricResult.metrics.accuracy;
                } else if (metricName === 'accuracy' && metricResult.accuracy !== undefined) {
                    value = metricResult.accuracy.value;
                } else if (metricName === 'performance' && metricResult.latency !== undefined) {
                    value = metricResult.latency.value;
                } else if (metricName === 'robustness' && metricResult.accuracy !== undefined) {
                    value = metricResult.accuracy.value;
                }
                
                if (value !== null) {
                    metricsData.push({
                        x: timestamp,
                        y: value,
                        runId: result.workflow_id
                    });
                }
            }
        });
        
        return metricsData.sort((a, b) => a.x - b.x);
    }

    /**
     * Filter workflow results by date range
     */
    filterByDateRange(dateRange) {
        this.filterCriteria.dateRange = dateRange;
        
        const now = new Date();
        let cutoffDate;
        
        switch (dateRange) {
            case '7d':
                cutoffDate = new Date(now.setDate(now.getDate() - 7));
                break;
            case '1m':
                cutoffDate = new Date(now.setMonth(now.getMonth() - 1));
                break;
            case '3m':
                cutoffDate = new Date(now.setMonth(now.getMonth() - 3));
                break;
            case 'all':
            default:
                return this.workflowResults;
        }
        
        return this.workflowResults.filter(result => {
            const resultDate = new Date(result.timestamp);
            return resultDate >= cutoffDate;
        });
    }

    /**
     * Get dashboard stats from the workflow results
     */
    getDashboardStats() {
        const totalEvaluations = this.workflowResults.length;
        
        let completedTests = 0;
        this.workflowResults.forEach(result => {
            if (result.evaluation_status) {
                const statuses = Object.values(result.evaluation_status);
                if (statuses.every(status => status === 'completed')) {
                    completedTests++;
                }
            }
        });
        
        const latestRun = this.workflowResults.length > 0 ? 
            new Date(this.workflowResults[0].timestamp).toLocaleString() : '-';
        
        const successRate = totalEvaluations > 0 ? 
            Math.round((completedTests / totalEvaluations) * 100) : 0;
        
        return {
            totalEvaluations,
            completedTests,
            latestRun,
            successRate
        };
    }

    /**
     * Get the difference between two runs
     */
    getRunDifference(currentRunId) {
        const currentRun = this.getWorkflowResult(currentRunId);
        if (!currentRun || !currentRun.evaluation_results) return null;
        
        const differences = {};
        
        // Helper function to extract value from different metric structures
        const extractMetricValue = (metricData) => {
            if (!metricData) return null;
            
            // Case 1: Direct value
            if (typeof metricData === 'number') return metricData;
            
            // Case 2: {value: number} format
            if (metricData.value !== undefined) return metricData.value;
            
            // Case 3: {accuracy: {value: number}} format
            if (metricData.accuracy && metricData.accuracy.value !== undefined) {
                return metricData.accuracy.value;
            }
            
            // Case 4: {metrics: {accuracy: number}} format
            if (metricData.metrics && metricData.metrics.accuracy !== undefined) {
                return metricData.metrics.accuracy;
            }
            
            // Case 5: {latency: {value: number}} format for performance metrics
            if (metricData.latency && metricData.latency.value !== undefined) {
                return metricData.latency.value;
            }
            
            // Case 6: For gradcam_coverage or other explainability metrics
            if (metricData.gradcam_coverage && metricData.gradcam_coverage.value !== undefined) {
                return metricData.gradcam_coverage.value;
            }
            
            // Case 7: For shap_importance or other explainability metrics
            if (metricData.shap_importance && metricData.shap_importance.value !== undefined) {
                return metricData.shap_importance.value;
            }
            
            return null;
        };
        
        // Sort runs by timestamp (newest first)
        const sortedRuns = [...this.workflowResults].sort((a, b) => {
            return new Date(b.timestamp) - new Date(a.timestamp);
        });
        
        // Find the current run's index
        const currentIndex = sortedRuns.findIndex(r => r.workflow_id === currentRunId);
        if (currentIndex === -1) return null;
        
        // For each metric in the current run, find the most recent previous run that has the same metric
        Object.keys(currentRun.evaluation_results).forEach(metricName => {
            const currentMetric = currentRun.evaluation_results[metricName];
            const currentValue = extractMetricValue(currentMetric);
            
            if (currentValue === null || isNaN(currentValue)) return;
            
            // Look for the most recent previous run with this metric
            for (let i = currentIndex + 1; i < sortedRuns.length; i++) {
                const previousRun = sortedRuns[i];
                
                if (previousRun.evaluation_results && 
                    previousRun.evaluation_results[metricName]) {
                    
                    const previousMetric = previousRun.evaluation_results[metricName];
                    const previousValue = extractMetricValue(previousMetric);
                    
                    if (previousValue !== null && !isNaN(previousValue) && previousValue !== 0) {
                        // We found a valid previous value for this metric
                        differences[metricName] = {
                            from: previousValue,
                            to: currentValue,
                            change: currentValue - previousValue,
                            percentChange: ((currentValue - previousValue) / Math.abs(previousValue)) * 100,
                            previousRunId: previousRun.workflow_id,
                            previousRunTimestamp: previousRun.timestamp
                        };
                        
                        // Found a match, stop looking for this metric
                        break;
                    }
                }
            }
        });
        
        return {
            currentRun,
            differences
        };
    }

    /**
     * Get mock data for demonstration purposes
     */
    getMockWorkflowResults() {
        return [
            {
                workflow_id: "dcea2f",
                commit_hash: "b47a0db86a3f1f1ed010cb1b14ccce5a05d1318b",
                workflow_stage: "evaluation_selection",
                timestamp: "2025-05-13T16:07:06.878907",
                required_metrics: {
                    accuracy: "To assess the overall classification accuracy after architectural changes and weight initialization modifications.",
                    f1: "To evaluate the model's harmonic mean of precision and recall, which is important for binary classification.",
                    latency: "To measure the model's inference time, especially after alterations to the convolutional layers and feature extraction process."
                },
                optional_metrics: {
                    explainability: "To understand how the model's decision-making process may have changed due to modifications in the architecture and feature extraction mechanisms.",
                    memory_usage: "To track any changes in model memory consumption after adding an intermediate fully connected layer and adjusting weight initialization methods.",
                    robustness: "To evaluate the stability of the model under perturbations, especially with changes in weight initialization and dropout rates."
                },
                evaluation_status: {
                    accuracy: "completed",
                    performance: "completed",
                    robustness: "completed",
                    explainability: "completed"
                },
                evaluation_results: {
                    accuracy: {
                        accuracy: {
                            value: 0.9912,
                            total_samples: 10000
                        },
                        class_metrics: [
                            {
                                class: 0,
                                accuracy: 0.9959183673469387,
                                samples: 980
                            },
                            {
                                class: 1,
                                accuracy: 0.9938325991189427,
                                samples: 1135
                            }
                        ]
                    },
                    performance: {
                        latency: {
                            value: 0.0038992069005966187,
                            unit: "seconds",
                            total_samples: 10000
                        },
                        throughput: {
                            value: 249.49219615648664,
                            unit: "samples/second",
                            total_samples: 10000
                        }
                    },
                    robustness: {
                        accuracy: {
                            value: 0.9912,
                            unit: "ratio",
                            total_samples: 10000
                        },
                        confidence: {
                            value: 0.9932654378890992,
                            unit: "ratio",
                            total_samples: 10000
                        }
                    }
                },
                messages: [
                    {
                        agent: "evaluator",
                        type: "evaluation_complete",
                        content: "Evaluation accuracy completed successfully",
                        evaluation: "accuracy",
                        timestamp: "2025-05-13T16:07:53.508323"
                    },
                    {
                        agent: "evaluator",
                        type: "evaluation_complete",
                        content: "Evaluation performance completed successfully",
                        evaluation: "performance",
                        timestamp: "2025-05-13T16:08:33.602891"
                    }
                ]
            },
            {
                workflow_id: "ae98b1",
                commit_hash: "df7898da8a91f1dcbbc4803ebcb17a6f2cc5e66b",
                workflow_stage: "completion",
                timestamp: "2025-05-06T20:32:51.707719",
                required_metrics: {
                    accuracy: "Required to verify model output correctness",
                    performance: "Required to measure computational efficiency"
                },
                optional_metrics: {
                    robustness: "Optional evaluation for model stability"
                },
                evaluation_status: {
                    robustness: "completed",
                    accuracy: "completed",
                    performance: "completed"
                },
                evaluation_results: {
                    robustness: {
                        status: "COMPLETED",
                        metrics: {
                            clean_accuracy: 0.991,
                            noisy_accuracy: 0.992,
                            prediction_stability: 0.995
                        }
                    },
                    accuracy: {
                        status: "COMPLETED",
                        metrics: {
                            accuracy: 0.991,
                            precision: 0.991149423853153,
                            recall: 0.991,
                            f1_score: 0.9910108828228823
                        }
                    },
                    performance: {
                        status: "COMPLETED",
                        metrics: {
                            latency_ms: 24.804681539535522,
                            throughput: 1259.8428224200925,
                            memory_mb: 0.0
                        }
                    }
                }
            }
        ];
    }

    /**
     * Get net change data for charting
     */
    getNetChangeData(metricName) {
        const changeData = [];
        
        // Go through workflow results chronologically
        const sortedResults = [...this.workflowResults].sort((a, b) => {
            return new Date(a.timestamp) - new Date(b.timestamp);
        });
        
        // Calculate change between consecutive runs
        for (let i = 1; i < sortedResults.length; i++) {
            const currentRun = sortedResults[i];
            const previousRun = sortedResults[i - 1];
            
            // Skip if either doesn't have evaluation results for the metric
            if (!currentRun.evaluation_results || !previousRun.evaluation_results ||
                !currentRun.evaluation_results[metricName] || !previousRun.evaluation_results[metricName]) {
                continue;
            }
            
            // Get values for comparison
            let currentValue = null;
            let previousValue = null;
            
            // Extract values based on metric data structure
            const currentMetric = currentRun.evaluation_results[metricName];
            const previousMetric = previousRun.evaluation_results[metricName];
            
            // Different formats observed in the data
            if (currentMetric.value !== undefined && previousMetric.value !== undefined) {
                currentValue = currentMetric.value;
                previousValue = previousMetric.value;
            } else if (currentMetric.metrics && previousMetric.metrics && 
                       currentMetric.metrics.accuracy !== undefined && previousMetric.metrics.accuracy !== undefined) {
                currentValue = currentMetric.metrics.accuracy;
                previousValue = previousMetric.metrics.accuracy;
            } else if (metricName === 'accuracy' && 
                       currentMetric.accuracy && previousMetric.accuracy && 
                       currentMetric.accuracy.value !== undefined && previousMetric.accuracy.value !== undefined) {
                currentValue = currentMetric.accuracy.value;
                previousValue = previousMetric.accuracy.value;
            } else if (metricName === 'performance' && 
                       currentMetric.latency && previousMetric.latency && 
                       currentMetric.latency.value !== undefined && previousMetric.latency.value !== undefined) {
                currentValue = currentMetric.latency.value;
                previousValue = previousMetric.latency.value;
            } else if (metricName === 'robustness' && 
                       currentMetric.accuracy && previousMetric.accuracy && 
                       currentMetric.accuracy.value !== undefined && previousMetric.accuracy.value !== undefined) {
                currentValue = currentMetric.accuracy.value;
                previousValue = previousMetric.accuracy.value;
            }
            
            // Calculate change if we have both values
            if (currentValue !== null && previousValue !== null) {
                const change = currentValue - previousValue;
                
                changeData.push({
                    x: new Date(currentRun.timestamp),
                    y: change,
                    runId: currentRun.workflow_id,
                    previousRunId: previousRun.workflow_id
                });
            }
        }
        
        return changeData;
    }
}

// Create a singleton instance
const dataService = new DataService();

export default dataService; 