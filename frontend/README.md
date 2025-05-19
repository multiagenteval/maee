# MAEE Evaluation Dashboard

A professional dark-mode dashboard for visualizing machine learning evaluation results over time from the MAEE evaluation ecosystem.

## Features

- **Timeline View**: Track test runs chronologically with detailed statuses
- **Metrics Visualization**: Plot key metrics over time with customizable time ranges
- **Detailed Run Analysis**: View in-depth metrics, code changes, and messages for each run
- **Class-specific Metrics**: Visualize per-class performance for classification tasks
- **Code Diff Viewer**: See exactly what changed between commits

## Structure

- **index.html**: Main dashboard HTML file
- **styles/main.css**: Dark mode professional styling
- **scripts/**
  - **main.js**: Core application logic and UI interactions
  - **data.js**: Data processing and management
  - **charts.js**: Chart visualizations using Chart.js
  - **utils.js**: Helper functions for formatting and display

## Getting Started

1. Clone the repository
2. Navigate to the frontend directory
3. Set up a local server (e.g., `python -m http.server` or use an extension like Live Server in VSCode)
4. Open the browser at the local server address

## Dependencies

- **Chart.js**: For metrics visualization
- **Luxon**: For date formatting and handling
- **Inter font**: For typography

## Technical Decisions

### Dark Mode Design

The dark mode design was chosen for its professional appearance and reduced eye strain for ML engineers who may be monitoring evaluations over long periods.

### Architecture

The frontend uses a clean separation of concerns:
- **Data Layer**: Handles fetching, parsing, and preparing data
- **Visualization Layer**: Creates and updates charts and visualizations
- **UI Layer**: Manages interactions and state

### Responsiveness

The dashboard is designed to work well on both desktop and tablet screens. Key components like charts and metric cards will adapt to different screen sizes.

## Future Enhancements

- **Real-time Updates**: Add WebSocket support for live updates
- **More Chart Types**: Add additional visualizations (confusion matrices, ROC curves, etc.)
- **Export Functionality**: Allow exporting reports as PDF or images
- **Custom Dashboards**: Allow users to create custom dashboard layouts

## License

MIT 