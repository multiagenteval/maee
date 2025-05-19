/**
 * Utility functions for the MAEE Evaluation Dashboard
 */

/**
 * Format a date for display
 */
export function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleString();
}

/**
 * Format a number with appropriate precision
 */
export function formatNumber(number, precision = 4) {
    if (number === undefined || number === null) return '-';
    
    // If it's a very small number, show more precision
    if (Math.abs(number) < 0.01) {
        return number.toExponential(precision);
    }
    
    return number.toFixed(precision);
}

/**
 * Create an element with attributes and children
 */
export function createElement(tag, attributes = {}, children = []) {
    const element = document.createElement(tag);
    
    // Set attributes
    Object.entries(attributes).forEach(([key, value]) => {
        if (key === 'className') {
            element.className = value;
        } else if (key === 'textContent') {
            element.textContent = value;
        } else if (key === 'innerText') {
            element.innerText = value;
        } else if (key === 'innerHTML') {
            element.innerHTML = value;
        } else if (key.startsWith('on') && typeof value === 'function') {
            element.addEventListener(key.substring(2).toLowerCase(), value);
        } else {
            element.setAttribute(key, value);
        }
    });
    
    // Append children
    children.forEach(child => {
        if (typeof child === 'string') {
            element.appendChild(document.createTextNode(child));
        } else if (child instanceof Node) {
            element.appendChild(child);
        }
    });
    
    return element;
}

/**
 * Format a metric name for display
 */
export function formatMetricName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/([A-Z])/g, ' $1')
        .replace(/^./, str => str.toUpperCase())
        .trim();
}

/**
 * Get a status color class based on status
 */
export function getStatusClass(status) {
    status = status.toLowerCase();
    if (status === 'completed' || status === 'complete' || status === 'done' || status === 'success') {
        return 'completed';
    } else if (status === 'pending' || status === 'in_progress' || status === 'running') {
        return 'pending';
    } else {
        return 'error';
    }
}

/**
 * Truncate a string to a specific length
 */
export function truncate(str, length = 40) {
    if (!str) return '';
    return str.length > length ? str.substring(0, length) + '...' : str;
}

/**
 * Format code diff for display
 */
export function formatDiff(diff) {
    if (!diff) return '';
    
    let html = '';
    
    // Process diff lines
    const lines = diff.split('\n');
    lines.forEach(line => {
        if (line.startsWith('+')) {
            html += `<div class="diff-line diff-add">${escapeHtml(line)}</div>`;
        } else if (line.startsWith('-')) {
            html += `<div class="diff-line diff-remove">${escapeHtml(line)}</div>`;
        } else {
            html += `<div class="diff-line">${escapeHtml(line)}</div>`;
        }
    });
    
    return html;
}

/**
 * Escape HTML special characters
 */
export function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

/**
 * Get percentage change between two numbers with appropriate formatting
 */
export function getPercentageChange(from, to) {
    if (from === 0) return '—';
    
    const change = ((to - from) / Math.abs(from)) * 100;
    const prefix = change > 0 ? '+' : '';
    
    return `${prefix}${change.toFixed(2)}%`;
} 