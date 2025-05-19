/**
 * Main entry point for the MAEE Evaluation Dashboard
 */

import dataService from './data.js';
import chartService from './charts.js';
import * as utils from './utils.js';

class App {
    constructor() {
        this.currentMetric = 'accuracy';
        this.currentDateRange = '7d';
        this.selectedRunId = null;
        
        // DOM elements
        this.elements = {
            timelineContainer: document.getElementById('timelineContainer'),
            totalEvaluations: document.getElementById('totalEvaluations'),
            completedTests: document.getElementById('completedTests'),
            latestRun: document.getElementById('latestRun'),
            successRate: document.getElementById('successRate'),
            runDetails: document.getElementById('runDetails'),
            runModal: document.getElementById('runModal'),
            closeModalBtn: document.querySelector('.close-btn'),
            tabButtons: document.querySelectorAll('.tab-btn'),
            refreshBtn: document.getElementById('refreshBtn')
        };
        
        this.bindEvents();
    }
    
    /**
     * Initialize the application
     */
    async init() {
        try {
            // Load workflow results
            await dataService.loadWorkflowResults();
            
            // Initialize timeline
            this.initTimeline();
            
            // Initialize dashboard stats
            this.updateDashboardStats();
            
            console.log('App initialized successfully');
        } catch (error) {
            console.error('Error initializing app:', error);
        }
    }
    
    /**
     * Bind event listeners
     */
    bindEvents() {
        // Close modal
        this.elements.closeModalBtn.addEventListener('click', () => {
            this.elements.runModal.style.display = 'none';
        });
        
        // Tab switching
        this.elements.tabButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                // Update active tab button
                this.elements.tabButtons.forEach(b => b.classList.remove('active'));
                e.currentTarget.classList.add('active');
                
                // Show active tab content
                const tabId = e.currentTarget.dataset.tab;
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                document.getElementById(tabId).classList.add('active');
            });
        });
        
        // Refresh button
        this.elements.refreshBtn.addEventListener('click', () => {
            this.init();
        });
        
        // Close modal when clicking outside
        window.addEventListener('click', (e) => {
            if (e.target === this.elements.runModal) {
                this.elements.runModal.style.display = 'none';
            }
        });
    }
    
    /**
     * Initialize the timeline
     */
    initTimeline() {
        this.elements.timelineContainer.innerHTML = '';
        
        const results = dataService.workflowResults;
        
        results.forEach(result => {
            const timelineItem = this.createTimelineItem(result);
            this.elements.timelineContainer.appendChild(timelineItem);
        });
    }
    
    /**
     * Create a timeline item
     */
    createTimelineItem(result) {
        const timelineItem = document.createElement('div');
        timelineItem.className = 'timeline-item';
        timelineItem.dataset.id = result.workflow_id;
        
        const header = document.createElement('div');
        header.className = 'timeline-header';
        
        const id = document.createElement('div');
        id.className = 'timeline-id';
        id.textContent = result.workflow_id;
        
        const timestamp = document.createElement('div');
        timestamp.className = 'timeline-timestamp';
        timestamp.textContent = utils.formatDate(result.timestamp);
        
        header.appendChild(id);
        header.appendChild(timestamp);
        
        const metrics = document.createElement('div');
        metrics.className = 'timeline-metrics';
        
        // Add metrics badges
        if (result.evaluation_status) {
            Object.entries(result.evaluation_status).forEach(([metric, status]) => {
                const metricBadge = document.createElement('div');
                metricBadge.className = `timeline-metric ${utils.getStatusClass(status)}`;
                metricBadge.textContent = utils.formatMetricName(metric);
                metrics.appendChild(metricBadge);
            });
        }
        
        timelineItem.appendChild(header);
        timelineItem.appendChild(metrics);
        
        // Handle click on timeline item
        timelineItem.addEventListener('click', () => {
            // Update selection
            this.selectRun(result.workflow_id);
        });
        
        return timelineItem;
    }
    
    /**
     * Update the dashboard stats
     */
    updateDashboardStats() {
        const stats = dataService.getDashboardStats();
        
        this.elements.totalEvaluations.textContent = stats.totalEvaluations;
        this.elements.completedTests.textContent = stats.completedTests;
        this.elements.latestRun.textContent = stats.latestRun;
        this.elements.successRate.textContent = `${stats.successRate}%`;
    }
    
    /**
     * Update charts with current data
     */
    updateChart() {
        // Chart was removed from the main page
        // No charts to update
    }
    
    /**
     * Select a run and show its details
     */
    selectRun(runId) {
        this.selectedRunId = runId;
        
        // Update timeline items
        const timelineItems = this.elements.timelineContainer.querySelectorAll('.timeline-item');
        timelineItems.forEach(item => {
            item.classList.toggle('active', item.dataset.id === runId);
        });
        
        // Get run data
        const run = dataService.getWorkflowResult(runId);
        if (!run) return;
        
        // Hide the no-selection message and show the content
        const noSelection = document.querySelector('#runDetails .no-selection');
        const runDetailsContent = document.querySelector('#runDetails .run-details-content');
        
        if (noSelection && runDetailsContent) {
            noSelection.style.display = 'none';
            runDetailsContent.style.display = 'block';
        }
        
        // Update main view details
        this.updateMainViewDetails(run);
        
        // Prepare modal details
        this.prepareModalDetails(run);
    }
    
    /**
     * Update the main view with run details
     */
    updateMainViewDetails(run) {
        // Set basic info in main view
        document.getElementById('mainRunId').textContent = run.workflow_id;
        document.getElementById('mainRunTimestamp').textContent = utils.formatDate(run.timestamp);
        document.getElementById('mainRunCommit').textContent = utils.truncate(run.commit_hash, 10);
        document.getElementById('mainRunStage').textContent = run.workflow_stage || 'Unknown';
        
        // Set analysis in main view
        const mainRunAnalysis = document.getElementById('mainRunAnalysis');
        if (run.analysis) {
            mainRunAnalysis.textContent = run.analysis;
        } else if (run.openai_analysis) {
            if (typeof run.openai_analysis === 'string') {
                mainRunAnalysis.textContent = run.openai_analysis;
            } else {
                try {
                    mainRunAnalysis.textContent = JSON.stringify(run.openai_analysis, null, 2);
                    mainRunAnalysis.classList.add('formatted-analysis');
                } catch (e) {
                    mainRunAnalysis.textContent = 'Error formatting OpenAI analysis';
                }
            }
        } else {
            mainRunAnalysis.textContent = 'No analysis available';
        }
        
        // Set up the net change section in main view
        const mainNetChangeContainer = document.getElementById('mainNetChangeContainer');
        mainNetChangeContainer.innerHTML = '';
        
        // Get differences with previous runs using the updated getRunDifference method
        const runDiff = dataService.getRunDifference(run.workflow_id);
        
        if (runDiff && Object.keys(runDiff.differences).length > 0) {
            this.createNetChangeVisualization(mainNetChangeContainer, runDiff, this.currentMetric);
        } else {
            mainNetChangeContainer.innerHTML = '<div class="no-data">No comparable metrics found with previous runs</div>';
        }
        
        // Set up view details button
        const viewDetailsBtn = document.getElementById('viewDetailsBtn');
        if (viewDetailsBtn) {
            viewDetailsBtn.onclick = () => {
                this.elements.runModal.style.display = 'block';
            };
        }
    }
    
    /**
     * Prepare modal details but don't show the modal
     */
    prepareModalDetails(run) {
        // Set basic info
        document.getElementById('runId').textContent = run.workflow_id;
        document.getElementById('runTimestamp').textContent = utils.formatDate(run.timestamp);
        document.getElementById('runCommit').textContent = utils.truncate(run.commit_hash, 10);
        document.getElementById('runStage').textContent = run.workflow_stage || 'Unknown';
        
        // Set analysis
        const runAnalysis = document.getElementById('runAnalysis');
        if (run.analysis) {
            runAnalysis.textContent = run.analysis;
        } else if (run.openai_analysis) {
            if (typeof run.openai_analysis === 'string') {
                runAnalysis.textContent = run.openai_analysis;
            } else if (typeof run.openai_analysis === 'object') {
                try {
                    // Format the JSON for display
                    runAnalysis.textContent = JSON.stringify(run.openai_analysis, null, 2);
                    runAnalysis.classList.add('formatted-analysis');
                } catch (e) {
                    runAnalysis.textContent = 'Error formatting OpenAI analysis';
                }
            } else {
                runAnalysis.textContent = 'No analysis available';
            }
        } else {
            runAnalysis.textContent = 'No analysis available';
        }
        
        // Set up the net change section
        const netChangeContainer = document.getElementById('netChangeContainer');
        netChangeContainer.innerHTML = '';
        
        // Get differences with previous runs using the updated getRunDifference method
        const runDiff = dataService.getRunDifference(run.workflow_id);
        
        if (runDiff && Object.keys(runDiff.differences).length > 0) {
            // Create metrics selection dropdown for the net change section
            const metricSelector = document.createElement('div');
            metricSelector.className = 'metric-selector';
            
            const metricLabel = document.createElement('label');
            metricLabel.textContent = 'Select metric: ';
            
            const metricSelect = document.createElement('select');
            metricSelect.className = 'select-control';
            
            // Add options for each metric with differences
            Object.keys(runDiff.differences).forEach(metricName => {
                const option = document.createElement('option');
                option.value = metricName;
                option.textContent = utils.formatMetricName(metricName);
                metricSelect.appendChild(option);
            });
            
            metricSelector.appendChild(metricLabel);
            metricSelector.appendChild(metricSelect);
            netChangeContainer.appendChild(metricSelector);
            
            // Create the change visualization container
            const changeVisualization = document.createElement('div');
            changeVisualization.className = 'change-visualization';
            netChangeContainer.appendChild(changeVisualization);
            
            // Function to update the visualization based on selected metric
            const updateChangeVisualization = () => {
                const selectedMetric = metricSelect.value;
                this.createMetricChangeVisualization(changeVisualization, runDiff.differences[selectedMetric]);
            };
            
            // Set up the event listener for metric selection
            metricSelect.addEventListener('change', updateChangeVisualization);
            
            // Initialize with the first metric
            if (metricSelect.options.length > 0) {
                updateChangeVisualization();
            }
        } else {
            netChangeContainer.innerHTML = '<div class="no-data">No comparable metrics found with previous runs</div>';
        }
        
        // Continue with the rest of the modal content setup...
        this.showRunEvaluationStatus(run);
        this.showRunMetricsContent(run);
        this.showRunDiffContent(run);
        this.showRunMessageContent(run);
    }
    
    /**
     * Create a net change visualization for a specific metric
     */
    createNetChangeVisualization(container, runDiff, defaultMetric) {
        // If we have a metric that matches the current selected metric, use it
        let metricName = defaultMetric;
        
        // If the default metric isn't available in the differences, use the first available
        if (!runDiff.differences[metricName] && Object.keys(runDiff.differences).length > 0) {
            metricName = Object.keys(runDiff.differences)[0];
        }
        
        // If we found a valid metric to display
        if (runDiff.differences[metricName]) {
            // Create the metric selector
            const metricSelector = document.createElement('div');
            metricSelector.className = 'metric-selector';
            
            const metricLabel = document.createElement('label');
            metricLabel.textContent = 'Select metric: ';
            
            const metricSelect = document.createElement('select');
            metricSelect.className = 'select-control';
            
            // Add options for each metric with differences
            Object.keys(runDiff.differences).forEach(name => {
                const option = document.createElement('option');
                option.value = name;
                option.textContent = utils.formatMetricName(name);
                option.selected = name === metricName;
                metricSelect.appendChild(option);
            });
            
            metricSelector.appendChild(metricLabel);
            metricSelector.appendChild(metricSelect);
            container.appendChild(metricSelector);
            
            // Create the change visualization container
            const changeVisualization = document.createElement('div');
            changeVisualization.className = 'change-visualization';
            container.appendChild(changeVisualization);
            
            // Create the initial visualization
            this.createMetricChangeVisualization(changeVisualization, runDiff.differences[metricName]);
            
            // Set up the event listener for metric selection
            metricSelect.addEventListener('change', () => {
                const selectedMetric = metricSelect.value;
                this.createMetricChangeVisualization(changeVisualization, runDiff.differences[selectedMetric]);
            });
        }
    }
    
    /**
     * Create a visualization for a specific metric difference
     */
    createMetricChangeVisualization(container, metricDiff) {
        if (!metricDiff) return;
        
        container.innerHTML = '';
        
        // Create comparison display
        const comparisonContainer = document.createElement('div');
        comparisonContainer.className = 'comparison-container';
        
        // Previous value
        const prevValueContainer = document.createElement('div');
        prevValueContainer.className = 'comparison-item previous';
        
        const prevValueLabel = document.createElement('div');
        prevValueLabel.className = 'comparison-label';
        prevValueLabel.textContent = 'Previous';
        
        const prevValue = document.createElement('div');
        prevValue.className = 'comparison-value';
        prevValue.textContent = utils.formatNumber(metricDiff.from);
        
        prevValueContainer.appendChild(prevValueLabel);
        prevValueContainer.appendChild(prevValue);
        
        // Change indicator
        const changeContainer = document.createElement('div');
        changeContainer.className = 'comparison-item change';
        
        const changeArrow = document.createElement('div');
        changeArrow.className = 'change-arrow';
        changeArrow.innerHTML = metricDiff.change >= 0 ? '&#8593;' : '&#8595;'; // Up or down arrow
        
        const changeValue = document.createElement('div');
        changeValue.className = 'change-value';
        const changeClass = metricDiff.change >= 0 ? 'positive-change' : 'negative-change';
        changeValue.classList.add(changeClass);
        changeValue.textContent = utils.getPercentageChange(metricDiff.from, metricDiff.to);
        
        changeContainer.appendChild(changeArrow);
        changeContainer.appendChild(changeValue);
        
        // Current value
        const currValueContainer = document.createElement('div');
        currValueContainer.className = 'comparison-item current';
        
        const currValueLabel = document.createElement('div');
        currValueLabel.className = 'comparison-label';
        currValueLabel.textContent = 'Current';
        
        const currValue = document.createElement('div');
        currValue.className = 'comparison-value';
        currValue.textContent = utils.formatNumber(metricDiff.to);
        
        currValueContainer.appendChild(currValueLabel);
        currValueContainer.appendChild(currValue);
        
        // Add all components to the comparison container
        comparisonContainer.appendChild(prevValueContainer);
        comparisonContainer.appendChild(changeContainer);
        comparisonContainer.appendChild(currValueContainer);
        
        container.appendChild(comparisonContainer);
        
        // Add a sparkline chart to visualize the change
        const sparklineContainer = document.createElement('div');
        sparklineContainer.className = 'sparkline-container';
        sparklineContainer.style.height = '50px';
        
        container.appendChild(sparklineContainer);
        
        // Create a mini chart showing the change
        const canvas = document.createElement('canvas');
        sparklineContainer.appendChild(canvas);
        
        new Chart(canvas, {
            type: 'line',
            data: {
                labels: ['Previous', 'Current'],
                datasets: [{
                    data: [metricDiff.from, metricDiff.to],
                    borderColor: metricDiff.change >= 0 ? '#10b981' : '#ef4444',
                    backgroundColor: metricDiff.change >= 0 ? '#10b98133' : '#ef444433',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointBackgroundColor: metricDiff.change >= 0 ? '#10b981' : '#ef4444'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                },
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        display: false
                    }
                }
            }
        });
    }
    
    /**
     * Show run evaluation status
     */
    showRunEvaluationStatus(run) {
        const statusGrid = document.getElementById('evaluationStatus');
        statusGrid.innerHTML = '';
        
        if (run.evaluation_status) {
            Object.entries(run.evaluation_status).forEach(([metric, status]) => {
                const statusItem = document.createElement('div');
                statusItem.className = `status-item ${utils.getStatusClass(status)}`;
                
                const label = document.createElement('div');
                label.className = 'status-label';
                label.textContent = utils.formatMetricName(metric);
                
                const value = document.createElement('div');
                value.className = 'status-value';
                value.textContent = status;
                
                statusItem.appendChild(label);
                statusItem.appendChild(value);
                statusGrid.appendChild(statusItem);
            });
        } else {
            statusGrid.innerHTML = '<p>No evaluation status available</p>';
        }
    }
    
    /**
     * Show run metrics content
     */
    showRunMetricsContent(run) {
        const metricsContent = document.getElementById('metricsContent');
        metricsContent.innerHTML = '';
        
        if (run.evaluation_results) {
            Object.entries(run.evaluation_results).forEach(([metricName, results]) => {
                const metricsGroup = document.createElement('div');
                metricsGroup.className = 'metrics-group';
                
                const heading = document.createElement('h4');
                heading.textContent = utils.formatMetricName(metricName);
                metricsGroup.appendChild(heading);
                
                // Handle different metric result formats
                if (metricName === 'accuracy' && results.class_metrics) {
                    // Create table for accuracy metrics
                    const table = this.createMetricsTable(results, metricName);
                    metricsGroup.appendChild(table);
                    
                    // Create class metrics visualization
                    const classMetricsContainer = document.createElement('div');
                    metricsGroup.appendChild(classMetricsContainer);
                    chartService.createClassMetricsVisualization(classMetricsContainer, results.class_metrics);
                } else {
                    // Create table for generic metrics
                    const table = this.createMetricsTable(results, metricName);
                    metricsGroup.appendChild(table);
                }
                
                metricsContent.appendChild(metricsGroup);
            });
        } else {
            metricsContent.innerHTML = '<p>No evaluation results available</p>';
        }
    }
    
    /**
     * Show run diff content
     */
    showRunDiffContent(run) {
        const diffContent = document.getElementById('diffContent');
        diffContent.innerHTML = '';
        
        if (run.commit_diff) {
            // Add commit message with code formatting
            const commitMessage = document.createElement('div');
            commitMessage.className = 'diff-commit-message';
            
            // Create a heading for the commit message
            const commitHeading = document.createElement('h4');
            commitHeading.textContent = 'Commit Message';
            commitMessage.appendChild(commitHeading);
            
            // Format commit message as code
            if (run.commit_diff.message) {
                const messageContainer = document.createElement('div');
                messageContainer.className = 'commit-message-code';
                
                // Split commit message by lines for better formatting
                const messageLines = run.commit_diff.message.split('\n');
                
                // Create pre and code elements for proper code formatting
                const preElement = document.createElement('pre');
                const codeElement = document.createElement('code');
                
                // Process message lines with special formatting
                messageLines.forEach((line, index) => {
                    // Format the commit title (first line) differently
                    if (index === 0) {
                        const titleSpan = document.createElement('span');
                        titleSpan.className = 'commit-title';
                        titleSpan.textContent = line;
                        codeElement.appendChild(titleSpan);
                        codeElement.appendChild(document.createElement('br'));
                    } 
                    // Handle file changes in the commit message
                    else if (line.match(/^\s*[\w\/\.-]+\s*\|\s*\d+\s*[\+\-]*$/)) {
                        const fileSpan = document.createElement('span');
                        fileSpan.className = 'commit-file-change';
                        fileSpan.textContent = line;
                        codeElement.appendChild(fileSpan);
                        codeElement.appendChild(document.createElement('br'));
                    }
                    // Highlight sections starting with keywords like "feat:", "fix:", etc.
                    else if (line.match(/^\s*(feat|fix|docs|style|refactor|test|chore|perf|ci|build|revert)(\([\w-]+\))?:/i)) {
                        const keywordSpan = document.createElement('span');
                        keywordSpan.className = 'commit-keyword';
                        keywordSpan.textContent = line;
                        codeElement.appendChild(keywordSpan);
                        codeElement.appendChild(document.createElement('br'));
                    }
                    // Default formatting for regular lines
                    else {
                        const bodySpan = document.createElement('span');
                        bodySpan.className = 'commit-body';
                        bodySpan.textContent = line;
                        codeElement.appendChild(bodySpan);
                        codeElement.appendChild(document.createElement('br'));
                    }
                });
                
                preElement.appendChild(codeElement);
                messageContainer.appendChild(preElement);
                commitMessage.appendChild(messageContainer);
            } else {
                const noMessage = document.createElement('p');
                noMessage.textContent = 'No commit message available';
                commitMessage.appendChild(noMessage);
            }
            
            diffContent.appendChild(commitMessage);
            
            // Add file changes
            if (run.commit_diff.changes) {
                Object.entries(run.commit_diff.changes).forEach(([file, data]) => {
                    const diffFile = document.createElement('div');
                    diffFile.className = 'diff-file';
                    
                    const diffHeader = document.createElement('div');
                    diffHeader.className = 'diff-header';
                    diffHeader.textContent = file;
                    
                    const diffContentElement = document.createElement('div');
                    diffContentElement.className = 'diff-content';
                    
                    // Handle different formats
                    if (data.changes && Array.isArray(data.changes)) {
                        diffContentElement.innerHTML = utils.formatDiff(data.changes.join('\n'));
                    } else if (typeof data === 'string') {
                        diffContentElement.innerHTML = utils.formatDiff(data);
                    }
                    
                    diffFile.appendChild(diffHeader);
                    diffFile.appendChild(diffContentElement);
                    diffContent.appendChild(diffFile);
                });
            } else if (typeof run.commit_diff === 'string') {
                // Handle case where commit_diff is a string
                const diffContentElement = document.createElement('div');
                diffContentElement.className = 'diff-content';
                diffContentElement.innerHTML = utils.formatDiff(run.commit_diff);
                diffContent.appendChild(diffContentElement);
            }
        } else {
            diffContent.innerHTML = '<p>No commit diff available</p>';
        }
    }
    
    /**
     * Show run message content
     */
    showRunMessageContent(run) {
        const messageContent = document.getElementById('messageContent');
        messageContent.innerHTML = '';
        
        // Create the message list container
        const messageList = document.createElement('div');
        messageList.className = 'message-list';
        
        // Handle OpenAI analysis if it exists
        if (run.openai_analysis) {
            const openaiMessage = document.createElement('div');
            openaiMessage.className = 'message-item openai-analysis';
            
            const openaiHeader = document.createElement('div');
            openaiHeader.className = 'message-header';
            
            const openaiAgent = document.createElement('div');
            openaiAgent.className = 'message-agent';
            openaiAgent.textContent = 'OpenAI Analysis';
            
            const openaiTimestamp = document.createElement('div');
            openaiTimestamp.className = 'message-timestamp';
            openaiTimestamp.textContent = utils.formatDate(run.timestamp);
            
            openaiHeader.appendChild(openaiAgent);
            openaiHeader.appendChild(openaiTimestamp);
            
            const openaiContent = document.createElement('div');
            openaiContent.className = 'message-content';
            
            // Format OpenAI analysis content
            if (typeof run.openai_analysis === 'string') {
                openaiContent.textContent = run.openai_analysis;
            } else if (typeof run.openai_analysis === 'object') {
                try {
                    openaiContent.textContent = JSON.stringify(run.openai_analysis, null, 2);
                } catch (e) {
                    openaiContent.textContent = 'Error formatting OpenAI analysis';
                }
            }
            
            openaiMessage.appendChild(openaiHeader);
            openaiMessage.appendChild(openaiContent);
            messageList.appendChild(openaiMessage);
        }
        
        // Add regular messages
        if (run.messages && run.messages.length > 0) {
            run.messages.forEach(message => {
                const messageItem = document.createElement('div');
                messageItem.className = 'message-item';
                
                const messageHeader = document.createElement('div');
                messageHeader.className = 'message-header';
                
                const agent = document.createElement('div');
                agent.className = 'message-agent';
                agent.textContent = message.agent || 'System';
                
                const timestamp = document.createElement('div');
                timestamp.className = 'message-timestamp';
                timestamp.textContent = utils.formatDate(message.timestamp);
                
                messageHeader.appendChild(agent);
                messageHeader.appendChild(timestamp);
                
                const content = document.createElement('div');
                content.className = 'message-content';
                content.textContent = message.content || 'No content';
                
                messageItem.appendChild(messageHeader);
                messageItem.appendChild(content);
                messageList.appendChild(messageItem);
            });
            
            messageContent.appendChild(messageList);
        } else if (!run.openai_analysis) {
            // Only show this message if we don't have any messages at all (including OpenAI analysis)
            messageContent.innerHTML = '<p>No messages available</p>';
        } else {
            // We already added the OpenAI analysis to messageList
            messageContent.appendChild(messageList);
        }
    }
    
    /**
     * Show run details in the modal
     */
    showRunDetails(run) {
        // Prepare all the modal content
        this.prepareModalDetails(run);
        
        // Show modal
        this.elements.runModal.style.display = 'block';
    }
    
    /**
     * Create a metrics table from results
     */
    createMetricsTable(results, metricName) {
        const table = document.createElement('table');
        table.className = 'metrics-table';
        
        const thead = document.createElement('thead');
        const tbody = document.createElement('tbody');
        
        const headerRow = document.createElement('tr');
        headerRow.innerHTML = '<th>Metric</th><th>Value</th><th>Unit</th><th>Samples</th>';
        thead.appendChild(headerRow);
        
        // Flatten nested metric structure for display
        const flattenedMetrics = this.flattenMetrics(results, metricName);
        
        flattenedMetrics.forEach(metric => {
            const row = document.createElement('tr');
            
            const nameCell = document.createElement('td');
            nameCell.textContent = utils.formatMetricName(metric.name);
            
            const valueCell = document.createElement('td');
            valueCell.textContent = utils.formatNumber(metric.value);
            
            const unitCell = document.createElement('td');
            unitCell.textContent = metric.unit || '-';
            
            const samplesCell = document.createElement('td');
            samplesCell.textContent = metric.samples || '-';
            
            row.appendChild(nameCell);
            row.appendChild(valueCell);
            row.appendChild(unitCell);
            row.appendChild(samplesCell);
            
            tbody.appendChild(row);
        });
        
        table.appendChild(thead);
        table.appendChild(tbody);
        
        return table;
    }
    
    /**
     * Flatten nested metrics structure for display
     */
    flattenMetrics(results, parentName) {
        const metrics = [];
        
        if (typeof results !== 'object' || results === null) {
            return metrics;
        }
        
        // Special handling for different result formats
        if (results.metrics) {
            // Handle format with metrics object
            Object.entries(results.metrics).forEach(([name, value]) => {
                metrics.push({
                    name,
                    value,
                    unit: '-',
                    samples: results.total_samples || '-'
                });
            });
        } else if (results.value !== undefined) {
            // Handle format with value directly in the object
            metrics.push({
                name: parentName,
                value: results.value,
                unit: results.unit || '-',
                samples: results.total_samples || '-'
            });
        } else if (results.accuracy && results.accuracy.value !== undefined) {
            // Handle format with nested accuracy object
            metrics.push({
                name: 'accuracy',
                value: results.accuracy.value,
                unit: results.accuracy.unit || '-',
                samples: results.accuracy.total_samples || '-'
            });
            
            // Add other metrics at the same level
            Object.entries(results).forEach(([name, data]) => {
                if (name !== 'accuracy' && name !== 'class_metrics' && typeof data === 'object' && data !== null && data.value !== undefined) {
                    metrics.push({
                        name,
                        value: data.value,
                        unit: data.unit || '-',
                        samples: data.total_samples || '-'
                    });
                }
            });
        } else {
            // General case - traverse all properties
            Object.entries(results).forEach(([name, data]) => {
                if (name !== 'class_metrics') {
                    if (typeof data === 'object' && data !== null && !Array.isArray(data)) {
                        if (data.value !== undefined) {
                            metrics.push({
                                name,
                                value: data.value,
                                unit: data.unit || '-',
                                samples: data.total_samples || '-'
                            });
                        } else {
                            // Recursively flatten nested objects
                            metrics.push(...this.flattenMetrics(data, name));
                        }
                    } else if (typeof data === 'number') {
                        metrics.push({
                            name,
                            value: data,
                            unit: '-',
                            samples: '-'
                        });
                    }
                }
            });
        }
        
        return metrics;
    }
    
    /**
     * Set up click handlers for charts
     */
    setupChartClickHandlers() {
        // No charts on the main page anymore
    }
}

// Initialize application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const app = new App();
    app.init();
}); 