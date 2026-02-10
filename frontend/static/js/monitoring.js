// frontend/static/js/monitoring.js

document.addEventListener('DOMContentLoaded', function() {
    let queryChart = null;
    let responseChart = null;

    loadDashboard();

    function loadDashboard() {
        loadMetrics();
        loadRecentQueries();
        loadCharts();
    }

    async function loadMetrics() {
        try {
            const response = await fetch('/api/query/metrics');
            if (!response.ok) return;

            const data = await response.json();

            // Update metric cards
            document.getElementById('total-queries').textContent = data.total_queries || 0;
            document.getElementById('success-rate').textContent = `${data.success_rate || 0}%`;
            document.getElementById('avg-time').textContent = `${Math.round(data.avg_response_time_ms || 0)}ms`;
            document.getElementById('recent-24h').textContent = data.recent_24h || 0;

        } catch (error) {
            console.error('Failed to load metrics:', error);
        }
    }

    async function loadRecentQueries() {
        try {
            const response = await fetch('/api/query/history?limit=10');
            if (!response.ok) return;

            const data = await response.json();
            const queries = data.queries || [];

            const tbody = document.getElementById('recent-queries');

            if (queries.length === 0) {
                tbody.innerHTML = '<tr><td colspan="4" class="text-center text-muted">No queries yet</td></tr>';
                return;
            }

            let html = '';
            queries.forEach(q => {
                const statusBadge = q.success
                    ? '<span class="badge bg-success">Success</span>'
                    : '<span class="badge bg-danger">Error</span>';

                html += `
                    <tr>
                        <td>${escapeHtml(q.query.substring(0, 50))}${q.query.length > 50 ? '...' : ''}</td>
                        <td><code>${q.pipeline}</code></td>
                        <td>${statusBadge}</td>
                        <td class="text-muted small">${formatTimestamp(q.timestamp)}</td>
                    </tr>
                `;
            });

            tbody.innerHTML = html;

        } catch (error) {
            console.error('Failed to load recent queries:', error);
        }
    }

    function loadCharts() {
        // Query history chart (mock data for now)
        const ctx1 = document.getElementById('query-chart');
        if (ctx1 && !queryChart) {
            queryChart = new Chart(ctx1, {
                type: 'line',
                data: {
                    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    datasets: [{
                        label: 'Queries',
                        data: [12, 19, 15, 25, 22, 10, 8],
                        borderColor: '#0d6efd',
                        tension: 0.3,
                        fill: true,
                        backgroundColor: 'rgba(13, 110, 253, 0.1)'
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        y: { beginAtZero: true }
                    }
                }
            });
        }

        // Response time chart (mock data)
        const ctx2 = document.getElementById('response-chart');
        if (ctx2 && !responseChart) {
            responseChart = new Chart(ctx2, {
                type: 'doughnut',
                data: {
                    labels: ['< 1s', '1-2s', '2-5s', '> 5s'],
                    datasets: [{
                        data: [45, 30, 20, 5],
                        backgroundColor: ['#198754', '#0dcaf0', '#ffc107', '#dc3545']
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: 'bottom' }
                    }
                }
            });
        }
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function formatTimestamp(timestamp) {
        if (!timestamp) return '';
        const date = new Date(timestamp);
        return date.toLocaleString();
    }

    // Refresh every 30 seconds
    setInterval(loadDashboard, 30000);
});