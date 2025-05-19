/**
 * Charts for the MAEE Evaluation Dashboard
 */

class ChartService {
    constructor() {
        this.charts = {};
        this.chartColors = {
            accuracy: '#6366f1', // Indigo
            performance: '#06b6d4', // Cyan
            robustness: '#10b981', // Emerald
            f1: '#8b5cf6', // Violet
            memory_usage: '#ec4899', // Pink
            explainability: '#f59e0b', // Amber
            default: '#a0a0a0' // Gray
        };
    }

    /**
     * Initialize the metrics chart
     */
    initMetricsChart() {
        const ctx = document.getElementById('metricsChart').getContext('2d');
        
        this.charts.metrics = new Chart(ctx, {
            type: 'line',
            data: {
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    tooltip: {
                        backgroundColor: '#1e1e1e',
                        titleColor: '#ffffff',
                        bodyColor: '#a0a0a0',
                        borderColor: '#3a3a3a',
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 6,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += context.parsed.y.toFixed(4);
                                }
                                return label;
                            }
                        }
                    },
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#a0a0a0',
                            font: {
                                family: "'Inter', sans-serif",
                                size: 12
                            },
                            padding: 16
                        }
                    },
                    title: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            tooltipFormat: 'MMM d, yyyy HH:mm'
                        },
                        grid: {
                            color: 'rgba(58, 58, 58, 0.3)'
                        },
                        ticks: {
                            color: '#a0a0a0',
                            font: {
                                family: "'Inter', sans-serif",
                                size: 12
                            }
                        }
                    },
                    y: {
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(58, 58, 58, 0.3)'
                        },
                        ticks: {
                            color: '#a0a0a0',
                            font: {
                                family: "'Inter', sans-serif",
                                size: 12
                            }
                        }
                    }
                }
            }
        });
        
        return this.charts.metrics;
    }

    /**
     * Update the metrics chart with new data
     */
    updateMetricsChart(metricName, data) {
        if (!this.charts.metrics) {
            this.initMetricsChart();
        }
        
        // Clear existing datasets
        this.charts.metrics.data.datasets = [];
        
        // Add new dataset
        const color = this.chartColors[metricName] || this.chartColors.default;
        
        this.charts.metrics.data.datasets.push({
            label: this.formatMetricName(metricName),
            data: data,
            backgroundColor: `${color}33`, // Add transparency for the fill
            borderColor: color,
            borderWidth: 2,
            fill: true,
            tension: 0.3,
            pointRadius: 4,
            pointHoverRadius: 6,
            pointBackgroundColor: color,
            pointHoverBackgroundColor: '#ffffff',
            pointBorderColor: '#ffffff',
            pointHoverBorderColor: color,
            pointBorderWidth: 1,
            pointHoverBorderWidth: 2
        });
        
        // Adapt y-axis
        this.adaptYAxisScale(data.map(item => item.y));
        
        // Update chart
        this.charts.metrics.update();
    }

    /**
     * Adapt the y-axis scale based on the data
     */
    adaptYAxisScale(values) {
        if (values.length === 0) return;
        
        const min = Math.min(...values);
        const max = Math.max(...values);
        const range = max - min;
        
        // If the range is very small, expand it for better visibility
        if (range < 0.05) {
            let buffer = Math.max(0.01, range * 2);
            this.charts.metrics.options.scales.y.min = Math.max(0, min - buffer);
            this.charts.metrics.options.scales.y.max = max + buffer;
        } else {
            // If we have a reasonable range, add a small buffer at the top and bottom
            let buffer = range * 0.1;
            this.charts.metrics.options.scales.y.min = Math.max(0, min - buffer);
            this.charts.metrics.options.scales.y.max = max + buffer;
        }
    }
    
    /**
     * Format a metric name for display
     */
    formatMetricName(metricName) {
        // Convert snake_case or camelCase to Title Case
        return metricName
            .replace(/_/g, ' ')
            .replace(/([A-Z])/g, ' $1')
            .replace(/^./, str => str.toUpperCase())
            .trim();
    }

    /**
     * Create a class metrics chart for the accuracy metrics
     */
    createClassMetricsVisualization(container, classMetrics) {
        if (!classMetrics || !Array.isArray(classMetrics)) return;
        
        // Clear container
        container.innerHTML = '';
        
        // Create grid for all class metrics
        const grid = document.createElement('div');
        grid.className = 'class-metrics-grid';
        
        // Add each class metric
        classMetrics.forEach(metric => {
            const item = document.createElement('div');
            item.className = 'class-metric-item';
            
            const classNumber = document.createElement('div');
            classNumber.className = 'class-number';
            classNumber.textContent = metric.class;
            
            const accuracy = document.createElement('div');
            accuracy.className = 'class-accuracy';
            accuracy.textContent = (metric.accuracy * 100).toFixed(2) + '%';
            
            const samples = document.createElement('div');
            samples.className = 'class-samples';
            samples.textContent = `${metric.samples} samples`;
            
            const accuracyBar = document.createElement('div');
            accuracyBar.className = 'accuracy-bar';
            
            const accuracyValue = document.createElement('div');
            accuracyValue.className = 'accuracy-value';
            accuracyValue.style.width = `${metric.accuracy * 100}%`;
            
            accuracyBar.appendChild(accuracyValue);
            
            item.appendChild(classNumber);
            item.appendChild(accuracy);
            item.appendChild(samples);
            item.appendChild(accuracyBar);
            
            grid.appendChild(item);
        });
        
        container.appendChild(grid);
    }

    /**
     * Create and initialize the net change chart
     */
    initNetChangeChart() {
        const container = document.getElementById('netChangeChart');
        if (!container) return null;
        
        const ctx = container.getContext('2d');
        
        this.charts.netChange = new Chart(ctx, {
            type: 'bar',
            data: {
                datasets: []
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    tooltip: {
                        backgroundColor: '#1e1e1e',
                        titleColor: '#ffffff',
                        bodyColor: '#a0a0a0',
                        borderColor: '#3a3a3a',
                        borderWidth: 1,
                        padding: 12,
                        cornerRadius: 6,
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    const value = context.parsed.y;
                                    const prefix = value >= 0 ? '+' : '';
                                    return `${label}${prefix}${value.toFixed(4)}`;
                                }
                                return label;
                            }
                        }
                    },
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#a0a0a0',
                            font: {
                                family: "'Inter', sans-serif",
                                size: 12
                            },
                            padding: 16
                        }
                    },
                    title: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            tooltipFormat: 'MMM d, yyyy HH:mm'
                        },
                        grid: {
                            color: 'rgba(58, 58, 58, 0.3)'
                        },
                        ticks: {
                            color: '#a0a0a0',
                            font: {
                                family: "'Inter', sans-serif",
                                size: 12
                            }
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(58, 58, 58, 0.3)'
                        },
                        ticks: {
                            color: '#a0a0a0',
                            font: {
                                family: "'Inter', sans-serif",
                                size: 12
                            }
                        }
                    }
                }
            }
        });
        
        return this.charts.netChange;
    }

    /**
     * Update the net change chart with new data
     */
    updateNetChangeChart(metricName, changeData) {
        if (!this.charts.netChange) {
            this.initNetChangeChart();
        }
        
        if (!this.charts.netChange) return; // Exit if chart still doesn't exist
        
        // Clear existing datasets
        this.charts.netChange.data.datasets = [];
        
        // Add new dataset for net changes
        const color = this.chartColors[metricName] || this.chartColors.default;
        
        this.charts.netChange.data.datasets.push({
            label: `${this.formatMetricName(metricName)} Change`,
            data: changeData,
            backgroundColor: (context) => {
                const value = context.raw?.y;
                return value >= 0 ? '#10b98166' : '#ef444466'; // Green for positive, red for negative
            },
            borderColor: (context) => {
                const value = context.raw?.y;
                return value >= 0 ? '#10b981' : '#ef4444'; // Green for positive, red for negative
            },
            borderWidth: 1
        });
        
        // Update chart
        this.charts.netChange.update();
    }
}

// Create a singleton instance
const chartService = new ChartService();

export default chartService; 