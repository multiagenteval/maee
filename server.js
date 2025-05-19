const express = require('express');
const fs = require('fs');
const path = require('path');
const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files from the frontend directory
app.use(express.static('frontend'));

// Add JSON parsing middleware
app.use(express.json());

// API endpoint to get workflow results
app.get('/api/workflow-results', (req, res) => {
  try {
    // Read from the workflow directory
    const workflowDir = path.join(__dirname, 'output', 'workflow');
    const files = fs.readdirSync(workflowDir)
      .filter(file => file.startsWith('workflow_results_') && file.endsWith('.json'));
    
    // Read and parse each file
    const results = files.map(file => {
      const filePath = path.join(workflowDir, file);
      const content = fs.readFileSync(filePath, 'utf8');
      try {
        return JSON.parse(content);
      } catch (e) {
        console.error(`Error parsing ${file}:`, e);
        return null;
      }
    }).filter(result => result !== null);
    
    // Sort by timestamp (newest first)
    results.sort((a, b) => {
      const dateA = new Date(a.timestamp || 0);
      const dateB = new Date(b.timestamp || 0);
      return dateB - dateA;
    });
    
    res.json(results);
  } catch (error) {
    console.error('Error reading workflow results:', error);
    res.status(500).json({ error: 'Failed to get workflow results' });
  }
});

// Default route - serve the index.html
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'frontend', 'index.html'));
});

// Start the server
app.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}`);
  console.log(`Visit http://localhost:${PORT} to view the MAEE Evaluation Dashboard`);
}); 