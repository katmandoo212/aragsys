document.addEventListener('DOMContentLoaded', function() {
    const fetchForm = document.getElementById('fetch-form');
    const uploadForm = document.getElementById('upload-form');
    const refreshBtn = document.getElementById('refresh-btn');
    const documentList = document.getElementById('document-list');

    // Load documents on page load
    loadDocuments();

    // Refresh button
    refreshBtn.addEventListener('click', loadDocuments);

    async function loadDocuments() {
        documentList.innerHTML = '<p class="text-muted">Loading...</p>';

        try {
            const response = await fetch('/api/documents');
            if (!response.ok) throw new Error('Failed to load documents');

            const data = await response.json();
            displayDocuments(data.documents || []);
        } catch (error) {
            console.error('Load documents error:', error);
            documentList.innerHTML = '<p class="text-danger">Failed to load documents.</p>';
        }
    }

    function displayDocuments(documents) {
        if (documents.length === 0) {
            documentList.innerHTML = '<p class="text-muted">No documents indexed yet.</p>';
            return;
        }

        let html = '';
        documents.forEach(doc => {
            html += `
                <div class="document-item" id="doc-${doc.document_id}">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h6 class="mb-1">${doc.title}</h6>
                            <p class="mb-0 text-muted small">
                                <i class="bi bi-link-45deg"></i> ${doc.source}
                            </p>
                            <p class="mb-0 text-muted small">
                                <i class="bi bi-layers"></i> ${doc.chunk_count} chunks
                            </p>
                        </div>
                        <button class="btn btn-sm btn-outline-danger delete-btn" data-id="${doc.document_id}">
                            <i class="bi bi-trash"></i>
                        </button>
                    </div>
                </div>
            `;
        });

        documentList.innerHTML = html;

        // Add delete handlers
        document.querySelectorAll('.delete-btn').forEach(btn => {
            btn.addEventListener('click', async function() {
                const docId = this.dataset.id;
                if (confirm('Delete this document?')) {
                    await deleteDocument(docId);
                }
            });
        });
    }

    async function deleteDocument(docId) {
        try {
            const response = await fetch(`/api/documents/${docId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                // Remove from DOM
                const docElement = document.getElementById(`doc-${docId}`);
                if (docElement) {
                    docElement.remove();
                }
            } else {
                alert('Failed to delete document');
            }
        } catch (error) {
            console.error('Delete error:', error);
            alert('Failed to delete document');
        }
    }

    // Handle fetch form success
    fetchForm.addEventListener('htmx:afterRequest', function(evt) {
        if (evt.detail.xhr.status === 200) {
            setTimeout(loadDocuments, 500);
        }
    });

    // Handle upload form success
    uploadForm.addEventListener('htmx:afterRequest', function(evt) {
        if (evt.detail.xhr.status === 200) {
            document.getElementById('upload-file').value = '';
            setTimeout(loadDocuments, 500);
        }
    });
});