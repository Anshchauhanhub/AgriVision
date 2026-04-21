document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const analyzeBtn = document.getElementById('analyze-btn');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const dropZoneContent = document.querySelector('.drop-zone-content');
    const resultSection = document.getElementById('result-section');
    const resetBtn = document.getElementById('reset-btn');
    
    // Result elements
    const statusBadge = document.getElementById('status-badge');
    const resultPlant = document.getElementById('result-plant');
    const resultDisease = document.getElementById('result-disease');
    const resultConfidence = document.getElementById('result-confidence');
    const confidenceBar = document.getElementById('confidence-bar');

    let selectedFile = null;

    // Trigger file input
    dropZone.addEventListener('click', (e) => {
        if (e.target !== removeBtn && !removeBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    // Handle file selection
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag and drop handlers
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    ['dragleave', 'dragend'].forEach(type => {
        dropZone.addEventListener(type, () => {
            dropZone.classList.remove('drag-over');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            previewContainer.classList.remove('hidden');
            dropZoneContent.classList.add('hidden');
            analyzeBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    removeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        resetUpload();
    });

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        previewContainer.classList.add('hidden');
        dropZoneContent.classList.remove('hidden');
        analyzeBtn.disabled = true;
        resultSection.classList.add('hidden');
    }

    // Analyze button handler
    analyzeBtn.addEventListener('click', async () => {
        if (!selectedFile) return;

        const formData = new FormData();
        formData.append('file', selectedFile);

        // UI state: loading
        const btnText = analyzeBtn.querySelector('.btn-text');
        const loader = analyzeBtn.querySelector('.loader');
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        analyzeBtn.disabled = true;

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Analysis failed');

            const data = await response.json();
            showResults(data);
        } catch (error) {
            console.error(error);
            alert('Error analyzing image. Please try again.');
        } finally {
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
            analyzeBtn.disabled = false;
        }
    });

    function showResults(data) {
        resultSection.classList.remove('hidden');
        resultSection.scrollIntoView({ behavior: 'smooth' });

        // Update status badge
        statusBadge.textContent = data.status;
        statusBadge.className = 'badge ' + data.status.toLowerCase();

        // Update text content
        resultPlant.textContent = data.plant;
        resultDisease.textContent = data.disease;
        resultConfidence.textContent = data.confidence;

        // Animate confidence bar
        const confidenceVal = parseFloat(data.confidence);
        setTimeout(() => {
            confidenceBar.style.width = confidenceVal + '%';
        }, 100);
    }

    resetBtn.addEventListener('click', () => {
        resetUpload();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
});
