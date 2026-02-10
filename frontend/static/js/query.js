document.addEventListener('DOMContentLoaded', function() {
    const queryForm = document.getElementById('query-form');
    const queryInput = document.getElementById('query-input');
    const pipelineSelect = document.getElementById('pipeline-select');
    const submitBtn = document.getElementById('submit-btn');
    const progressContainer = document.getElementById('progress-container');
    const progressBar = document.getElementById('progress-bar');
    const progressMessage = document.getElementById('progress-message');
    const answerContainer = document.getElementById('answer-container');
    const answerMeta = document.getElementById('answer-meta');
    const retrievedDocs = document.getElementById('retrieved-docs');
    const copyBtn = document.getElementById('copy-btn');
    const citationModal = new bootstrap.Modal(document.getElementById('citation-modal'));

    let currentAnswer = '';

    queryForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        const query = queryInput.value.trim();
        const pipeline = pipelineSelect.value;

        if (!query) {
            alert('Please enter a question');
            return;
        }

        // Reset UI
        answerContainer.innerHTML = '<p class="text-muted">Processing...</p>';
        answerMeta.style.display = 'none';
        retrievedDocs.innerHTML = '<p class="text-muted">Retrieving documents...</p>';
        progressContainer.style.display = 'block';
        progressBar.style.width = '0%';
        progressMessage.textContent = 'Submitting query...';
        submitBtn.disabled = true;

        try {
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query, pipeline})
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            const taskId = data.task_id;

            // Subscribe to SSE stream
            const eventSource = new EventSource(`/api/query/stream/${taskId}`);

            eventSource.onmessage = function(event) {
                if (event.data === '[DONE]') {
                    eventSource.close();
                    progressContainer.style.display = 'none';
                    submitBtn.disabled = false;
                    return;
                }

                const eventData = JSON.parse(event.data);

                // Update progress
                if (eventData.progress) {
                    progressBar.style.width = `${eventData.progress}%`;
                }
                if (eventData.message) {
                    progressMessage.textContent = eventData.message;
                }

                // Handle complete
                if (eventData.status === 'complete' && eventData.data) {
                    displayAnswer(eventData.data);
                    progressContainer.style.display = 'none';
                    submitBtn.disabled = false;
                }

                // Handle error
                if (eventData.status === 'error') {
                    answerContainer.innerHTML = `<div class="alert alert-danger">Error: ${eventData.data?.error || 'Unknown error'}</div>`;
                    progressContainer.style.display = 'none';
                    submitBtn.disabled = false;
                }
            };

            eventSource.onerror = function() {
                eventSource.close();
                progressContainer.style.display = 'none';
                submitBtn.disabled = false;
                answerContainer.innerHTML = '<div class="alert alert-warning">Connection lost. Please try again.</div>';
            };

        } catch (error) {
            console.error('Query error:', error);
            answerContainer.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            progressContainer.style.display = 'none';
            submitBtn.disabled = false;
        }
    });

    function displayAnswer(data) {
        // Display answer
        let answerHtml = data.content;

        // Add citations as superscripts
        if (data.citations && data.citations.length > 0) {
            answerHtml += '<div class="mt-4"><h6>Citations</h6><ul>';
            data.citations.forEach((cite, index) => {
                answerHtml += `<li><a href="#" class="citation-link" data-source="${cite}" data-bs-toggle="modal" data-bs-target="#citation-modal">[${index + 1}] ${cite}</a></li>`;
            });
            answerHtml += '</ul></div>';
        }

        answerContainer.innerHTML = answerHtml;
        currentAnswer = data.content;

        // Show metadata
        document.getElementById('response-time').textContent = `${data.response_time_ms}ms`;
        document.getElementById('doc-count').textContent = data.retrieved_docs;
        answerMeta.style.display = 'block';
        copyBtn.style.display = 'inline-block';

        // Display retrieved documents (mock for now)
        retrievedDocs.innerHTML = `
            <div class="document-item">
                <strong>[1] sample1.pdf</strong>
                <p class="mb-0 text-muted small">Score: 0.85 • Page 1</p>
            </div>
            <div class="document-item">
                <strong>[2] sample2.md</strong>
                <p class="mb-0 text-muted small">Score: 0.78</p>
            </div>
        `;
    }

    // Copy button
    copyBtn.addEventListener('click', function() {
        navigator.clipboard.writeText(currentAnswer).then(() => {
            copyBtn.innerHTML = '<i class="bi bi-check"></i> Copied!';
            setTimeout(() => {
                copyBtn.innerHTML = '<i class="bi bi-clipboard"></i> Copy';
            }, 2000);
        });
    });

    // Citation modal
    document.querySelectorAll('.citation-link').forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const source = this.dataset.source;
            document.getElementById('citation-title').textContent = source;
            document.getElementById('citation-content').innerHTML = `
                <p><strong>Source:</strong> ${source}</p>
                <p><strong>Content:</strong></p>
                <p class="text-muted">This is the document content that was referenced in the answer.</p>
            `;
        });
    });
});