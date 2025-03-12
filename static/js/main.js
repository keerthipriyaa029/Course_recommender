// Main JavaScript file

// Handle card selection (for domain and skill selection pages)
document.addEventListener('DOMContentLoaded', function() {
    // Card selection functionality
    const selectableCards = document.querySelectorAll('.selectable-card');
    selectableCards.forEach(card => {
        card.addEventListener('click', function() {
            if (this.dataset.maxSelect === '1') {
                // Single selection mode
                selectableCards.forEach(c => c.classList.remove('selected'));
                this.classList.add('selected');
            } else {
                // Multiple selection mode
                this.classList.toggle('selected');
            }
        });
    });

    // Form validation
    const forms = document.querySelectorAll('.needs-validation');
    forms.forEach(form => {
        form.addEventListener('submit', function(event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });

    // Chatbot functionality
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');

    if (chatForm) {
        chatForm.addEventListener('submit', async function(event) {
            event.preventDefault();
            const message = chatInput.value.trim();
            if (!message) return;

            // Add user message to chat
            appendMessage('user', message);
            chatInput.value = '';

            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ message: message })
                });

                const data = await response.json();
                appendMessage('bot', data.response);
            } catch (error) {
                console.error('Error:', error);
                appendMessage('bot', 'Sorry, there was an error processing your request.');
            }
        });
    }
});

// Helper function to append messages in chat
function appendMessage(sender, message) {
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('chat-message', `${sender}-message`, 'fade-in');
    messageDiv.textContent = message;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Toast notification function
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.classList.add('toast', 'show', `bg-${type}`);
    toast.setAttribute('role', 'alert');
    toast.innerHTML = `
        <div class="toast-body text-white">
            ${message}
            <button type="button" class="btn-close btn-close-white float-end" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    const toastContainer = document.querySelector('.toast-container');
    toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.remove();
    }, 3000);
}

// Handle loading states
function setLoading(element, isLoading) {
    if (isLoading) {
        element.disabled = true;
        element.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Loading...';
    } else {
        element.disabled = false;
        element.innerHTML = element.getAttribute('data-original-text');
    }
}

// Initialize tooltips and popovers
var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
    return new bootstrap.Tooltip(tooltipTriggerEl);
});

var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
    return new bootstrap.Popover(popoverTriggerEl);
}); 