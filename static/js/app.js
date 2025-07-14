class RicePestDetector {
    constructor() {
        this.imageInput = document.getElementById('imageInput');
        this.originalCanvas = document.getElementById('originalCanvas');
        this.resultCanvas = document.getElementById('resultCanvas');
        this.detectBtn = document.getElementById('detectBtn');
        this.detectSahiBtn = document.getElementById('detectSahiBtn');
        this.sahiSettingsBtn = document.getElementById('sahiSettingsBtn');
        this.sahiConfig = document.getElementById('sahiConfig');
        this.detectionInfo = document.getElementById('detectionInfo');
        
        this.currentImage = null;
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        this.imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        this.detectBtn.addEventListener('click', () => this.runDetection(false));
        this.detectSahiBtn.addEventListener('click', () => this.runDetection(true));
        this.sahiSettingsBtn.addEventListener('click', () => this.toggleSahiSettings());
        
        // SAHI config sliders with null checks
        const sliceSize = document.getElementById('sliceSize');
        const sliceSizeValue = document.getElementById('sliceSizeValue');
        if (sliceSize && sliceSizeValue) {
            sliceSize.addEventListener('input', (e) => {
                sliceSizeValue.textContent = e.target.value;
            });
        }
        
        const overlap = document.getElementById('overlap');
        const overlapValue = document.getElementById('overlapValue');
        if (overlap && overlapValue) {
            overlap.addEventListener('input', (e) => {
                overlapValue.textContent = e.target.value;
            });
        }
        
        const confidenceThreshold = document.getElementById('confidenceThreshold');
        const confidenceValue = document.getElementById('confidenceValue');
        if (confidenceThreshold && confidenceValue) {
            confidenceThreshold.addEventListener('input', (e) => {
                confidenceValue.textContent = e.target.value;
            });
        }
        
        const nmsThreshold = document.getElementById('nmsThreshold');
        const nmsValue = document.getElementById('nmsValue');
        if (nmsThreshold && nmsValue) {
            nmsThreshold.addEventListener('input', (e) => {
                nmsValue.textContent = e.target.value;
            });
        }
    }
    
    handleImageUpload(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            this.currentImage = e.target.result;
            this.displayImage(this.currentImage, this.originalCanvas);
            this.detectBtn.disabled = false;
            this.detectSahiBtn.disabled = false;
            this.clearResults();
        };
        reader.readAsDataURL(file);
    }
    
    displayImage(imageSrc, canvas) {
        const img = new Image();
        img.onload = () => {
            canvas.width = img.width;
            canvas.height = img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0);
        };
        img.src = imageSrc;
    }
    
    async runDetection(useSahi = false) {
        if (!this.currentImage) return;
        
        const btn = useSahi ? this.detectSahiBtn : this.detectBtn;
        const originalText = btn.textContent;
        btn.textContent = '⏳ Processing...';
        btn.disabled = true;
        
        try {
            const twoStageToggle = document.getElementById('twoStageToggle');
            const payload = {
                image: this.currentImage,
                use_two_stage: twoStageToggle ? twoStageToggle.checked : false
            };
            
            // Add confidence threshold to payload
            const confidenceThresholdEl = document.getElementById('confidenceThreshold');
            payload.confidence_threshold = confidenceThresholdEl ? parseFloat(confidenceThresholdEl.value) : 0.4;
            
            if (useSahi) {
                const sliceSizeEl = document.getElementById('sliceSize');
                const overlapEl = document.getElementById('overlap');
                const nmsThresholdEl = document.getElementById('nmsThreshold');
                
                payload.sahi_config = {
                    slice_height: sliceSizeEl ? parseInt(sliceSizeEl.value) : 320,
                    slice_width: sliceSizeEl ? parseInt(sliceSizeEl.value) : 320,
                    overlap_height_ratio: overlapEl ? parseInt(overlapEl.value) / 100 : 0.2,
                    overlap_width_ratio: overlapEl ? parseInt(overlapEl.value) / 100 : 0.2,
                    nms_threshold: nmsThresholdEl ? parseFloat(nmsThresholdEl.value) : 0.5
                };
            }
            
            const endpoint = useSahi ? '/api/detect-sahi' : '/api/detect';
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            
            if (response.ok) {
                this.displayResults(result);
            } else {
                throw new Error(result.error || 'Detection failed');
            }
            
        } catch (error) {
            console.error('Detection error:', error);
            this.detectionInfo.innerHTML = `<div class="error">Error: ${error.message}</div>`;
        } finally {
            btn.textContent = originalText;
            btn.disabled = false;
        }
    }
    
    displayResults(result) {
        // Copy original image to result canvas
        this.displayImage(this.currentImage, this.resultCanvas);
        
        // Draw bounding boxes
        setTimeout(() => {
            this.drawBoundingBoxes(result.detections);
            this.showDetectionInfo(result);
        }, 100);
    }
    
    drawBoundingBoxes(detections) {
        const ctx = this.resultCanvas.getContext('2d');
        
        detections.forEach((detection, index) => {
            const [x1, y1, x2, y2] = detection.bbox;
            const width = x2 - x1;
            const height = y2 - y1;
            
            // Draw bounding box
            ctx.strokeStyle = detection.class === 'OS' ? '#ff6b6b' : '#4ecdc4';
            ctx.lineWidth = 3;
            ctx.strokeRect(x1, y1, width, height);
            
            // Draw label background
            const label = `${detection.class} ${(detection.confidence * 100).toFixed(1)}%`;
            ctx.font = '16px Arial';
            const textWidth = ctx.measureText(label).width;
            
            ctx.fillStyle = detection.class === 'OS' ? '#ff6b6b' : '#4ecdc4';
            ctx.fillRect(x1, y1 - 25, textWidth + 10, 25);
            
            // Draw label text
            ctx.fillStyle = 'white';
            ctx.fillText(label, x1 + 5, y1 - 5);
        });
    }
    
    showDetectionInfo(result) {
        const { detections, processing_time, image_size, method, slice_count, two_stage_enabled, pest_detections, nonpest_detections } = result;
        
        // Count occurrences of each class
        const classCounts = {};
        detections.forEach(det => {
            classCounts[det.class] = (classCounts[det.class] || 0) + 1;
        });
        
        const soCount = classCounts['SO'] || 0;
        const osCount = classCounts['OS'] || 0;
        
        let info = `
            <div class="detection-stats">
                <h4>Jumlah Deteksi</h4>
                <p><strong>Metode:</strong> ${method.toUpperCase()}</p>
                <p><strong>Waktu Proses:</strong> ${processing_time.toFixed(2)}s</p>
                <p><strong>Total Deteksi:</strong> ${detections.length}</p>
                ${two_stage_enabled ? '<p><strong>Two-Stage Detection:</strong> Aktif</p>' : ''}
        `;
        
        if (two_stage_enabled) {
            // When two-stage is enabled, show pest counts and nonpest counts as 'lain'
            const pestCount = pest_detections ? pest_detections.length : (soCount + osCount);
            const nonpestCount = nonpest_detections ? nonpest_detections.length : 0;
            info += `
                <p><strong>SO:</strong> ${soCount}</p>
                <p><strong>OS:</strong> ${osCount}</p>
                <p><strong>Lain:</strong> ${nonpestCount}</p>
            `;
        } else {
            // When two-stage is disabled, only show SO and OS counts
            info += `
                <p><strong>SO:</strong> ${soCount}</p>
                <p><strong>OS:</strong> ${osCount}</p>
            `;
        }
        
        info += `
                <p><strong>Ukuran Gambar:</strong> ${image_size[0]}×${image_size[1]}px</p>
                ${slice_count ? `<p><strong>Slices:</strong> ${slice_count}</p>` : ''}
            </div>
        `;
        
        this.detectionInfo.innerHTML = info;
    }
    
    clearResults() {
        const ctx = this.resultCanvas.getContext('2d');
        ctx.clearRect(0, 0, this.resultCanvas.width, this.resultCanvas.height);
        this.detectionInfo.innerHTML = '';
    }
    
    toggleSahiSettings() {
        const isHidden = this.sahiConfig.style.display === 'none';
        this.sahiConfig.style.display = isHidden ? 'block' : 'none';
        this.sahiSettingsBtn.textContent = isHidden ? '✕' : '⚙️';
    }
}

// Initialize the detector when page loads
document.addEventListener('DOMContentLoaded', () => {
    new RicePestDetector();
});