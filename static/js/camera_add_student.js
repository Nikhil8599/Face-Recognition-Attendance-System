document.addEventListener('DOMContentLoaded', function() {
    const saveInfoBtn = document.getElementById("saveInfoBtn");
    const startCameraBtn = document.getElementById("startCameraBtn");
    const stopCameraBtn = document.getElementById("stopCameraBtn");
    const captureBtn = document.getElementById("captureBtn");
    const startCaptureBtn = document.getElementById("startCaptureBtn");
    const addStudentBtn = document.getElementById("addStudentBtn");
    const video = document.getElementById("video");
    const cameraFrame = document.querySelector(".camera-frame");
    const capturePreview = document.getElementById("capturePreview");
    const captureStatus = document.getElementById("captureStatus");
    const captureCount = document.getElementById("captureCount");
    const progressBar = document.getElementById("progressBar");

    let stream = null;
    let capturedImages = [];
    let studentId = null;
    let isCapturing = false;
    const MAX_IMAGES = 20;
    updateStepIndicator(1);
    updateCaptureUI();
    function updateStepIndicator(step) {
        document.querySelectorAll('.step').forEach((el, index) => {
            el.classList.remove('active', 'completed');
            if (index + 1 < step) {
                el.classList.add('completed');
            } else if (index + 1 === step) {
                el.classList.add('active');
            }
        });
    }

    function updateCaptureUI() {
        const count = capturedImages.length;
        captureCount.textContent = `${count}/${MAX_IMAGES}`;
        progressBar.style.width = `${(count / MAX_IMAGES) * 100}%`;

        if (count >= MAX_IMAGES) {
            captureBtn.disabled = true;
            startCaptureBtn.disabled = true;
            addStudentBtn.disabled = false;
            captureStatus.innerHTML = `<span class="text-success"><i class="fas fa-check-circle"></i> ${MAX_IMAGES} images captured!</span>`;
        }
    }

    function showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');

        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };

        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <i class="fas fa-${icons[type] || 'info-circle'}"></i>
            <span>${message}</span>
        `;

        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    document.getElementById("studentForm").addEventListener("submit", async (e) => {
        e.preventDefault();
        const saveBtn = saveInfoBtn;
        saveBtn.disabled = true;
        saveBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Saving...';

        const formData = new FormData(e.target);

        try {
            const response = await fetch("/add_student", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                throw new Error('Failed to save student');
            }

            const data = await response.json();
            studentId = data.student_id;

            showToast(data.message || "Student saved successfully!", "success");
            saveBtn.innerHTML = '<i class="fas fa-check me-2"></i>Information Saved';
            startCameraBtn.disabled = false;
            startCaptureBtn.disabled = false;
            updateStepIndicator(2);

        } catch (error) {
            showToast(error.message || "Failed to save student information", "error");
            saveBtn.disabled = false;
            saveBtn.innerHTML = '<i class="fas fa-save me-2"></i>Save Student Information';
        }
    });

    startCameraBtn.addEventListener("click", async () => {
        startCameraBtn.disabled = true;
        startCameraBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Starting...';

        try {
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            };

            stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;

            video.onloadedmetadata = () => {
                video.play().then(() => {
                    cameraFrame.classList.add("camera-active");
                    captureBtn.disabled = false;
                    stopCameraBtn.disabled = false;
                    startCameraBtn.innerHTML = '<i class="fas fa-video me-2"></i>Camera Started';

                    captureStatus.innerHTML =
                        '<span class="text-success"><i class="fas fa-video"></i> Camera active - Ready to capture</span>';

                    showToast("Camera started successfully", "success");
                }).catch(err => {
                    showToast("Error playing video: " + err.message, "error");
                    stopCamera();
                });
            };

        } catch (error) {
            console.error("Camera error:", error);
            try {
                stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;

                video.onloadedmetadata = () => {
                    video.play().then(() => {
                        cameraFrame.classList.add("camera-active");
                        captureBtn.disabled = false;
                        stopCameraBtn.disabled = false;
                        startCameraBtn.innerHTML = '<i class="fas fa-video me-2"></i>Camera Started';

                        captureStatus.innerHTML =
                            '<span class="text-success"><i class="fas fa-video"></i> Camera active - Ready to capture</span>';

                        showToast("Camera started with default settings", "success");
                    });
                };

            } catch (fallbackError) {
                showToast("Camera access denied. Please allow camera permissions and refresh the page.", "error");
                startCameraBtn.disabled = false;
                startCameraBtn.innerHTML = '<i class="fas fa-video me-2"></i>Start Camera';
            }
        }
    });

    // Stop camera
    stopCameraBtn.addEventListener("click", () => {
        stopCamera();
    });

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            cameraFrame.classList.remove("camera-active");

            startCameraBtn.disabled = false;
            startCameraBtn.innerHTML = '<i class="fas fa-video me-2"></i>Start Camera';
            captureBtn.disabled = true;
            stopCameraBtn.disabled = true;

            captureStatus.innerHTML = '<span class="text-muted">Camera stopped</span>';

            showToast("Camera stopped", "info");
        }
    }

    // Manual capture
    captureBtn.addEventListener("click", async () => {
        if (!stream || capturedImages.length >= MAX_IMAGES || isCapturing) return;

        isCapturing = true;

        try {
            // Create canvas and capture frame
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth || 640;
            canvas.height = video.videoHeight || 480;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

            // Convert to blob
            const blob = await new Promise(resolve => {
                canvas.toBlob(resolve, 'image/jpeg', 0.9);
            });

            capturedImages.push(blob);

            // Add preview image
            const imgUrl = URL.createObjectURL(blob);
            const img = document.createElement('img');
            img.src = imgUrl;
            img.className = 'preview-image captured';
            img.title = `Image ${capturedImages.length}`;
            img.onload = () => URL.revokeObjectURL(imgUrl); // Clean up memory

            // Clear preview if we have more than 8 images
            if (capturedImages.length > 8) {
                capturePreview.innerHTML = '';
                capturedImages.forEach((blob, index) => {
                    const img = document.createElement('img');
                    img.src = URL.createObjectURL(blob);
                    img.className = 'preview-image captured';
                    img.title = `Image ${index + 1}`;
                    capturePreview.appendChild(img);
                });
            } else {
                capturePreview.appendChild(img);
            }

            // Update UI
            updateCaptureUI();

            // Visual feedback
            video.style.boxShadow = "0 0 0 5px #4CAF50";
            setTimeout(() => video.style.boxShadow = "none", 300);

            captureStatus.innerHTML =
                `<span class="text-info"><i class="fas fa-camera"></i> Captured ${capturedImages.length} of ${MAX_IMAGES} images</span>`;

            // Check if complete
            if (capturedImages.length === MAX_IMAGES) {
                showToast(`${MAX_IMAGES} images captured! Ready to upload.`, "success");
                updateStepIndicator(3);
            }

        } catch (error) {
            showToast("Capture failed: " + error.message, "error");
        } finally {
            isCapturing = false;
        }
    });

    // Auto capture
    startCaptureBtn.addEventListener("click", async () => {
        if (!stream) {
            showToast("Please start the camera first", "warning");
            return;
        }

        if (capturedImages.length >= MAX_IMAGES) {
            showToast(`Already captured ${MAX_IMAGES} images`, "info");
            return;
        }

        const btn = startCaptureBtn;
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Capturing...';
        captureBtn.disabled = true;

        try {
            // Auto capture remaining images
            const remaining = MAX_IMAGES - capturedImages.length;
            for (let i = 0; i < remaining; i++) {
                captureBtn.click();
                await new Promise(resolve => setTimeout(resolve, 800)); // 800ms delay between captures
            }

            btn.innerHTML = '<i class="fas fa-check me-2"></i>Auto Capture Complete';
            showToast("Auto capture completed!", "success");

        } catch (error) {
            showToast("Auto capture failed: " + error.message, "error");
            btn.disabled = false;
            captureBtn.disabled = false;
            btn.innerHTML = '<i class="fas fa-play-circle me-2"></i>Start Auto Capture';
        }
    });

    // Complete registration
    addStudentBtn.addEventListener("click", async () => {
        if (capturedImages.length === 0) {
            showToast("No images captured", "error");
            return;
        }

        const btn = addStudentBtn;
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Uploading...';

        try {
            // Upload images
            const formData = new FormData();
            formData.append("student_id", studentId);

            capturedImages.forEach((blob, index) => {
                formData.append("images[]", blob, `student_${studentId}_${Date.now()}_${index}.jpg`);
            });

            const response = await fetch("/upload_face", {
                method: "POST",
                body: formData
            });

            const result = await response.json();

            if (response.ok) {
                showToast(`Successfully uploaded ${result.saved} images for student ${studentId}`, "success");

                btn.innerHTML = '<i class="fas fa-check me-2"></i>Registration Complete';

                // Redirect after delay
                setTimeout(() => {
                    showToast("Returning to dashboard...", "info");
                    setTimeout(() => {
                        window.location.href = "/";
                    }, 1500);
                }, 2000);
            } else {
                throw new Error(result.error || "Upload failed");
            }

        } catch (error) {
            showToast("Upload failed: " + error.message, "error");
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-check-circle me-2"></i>Complete Student Registration';
        }
    });
    window.addEventListener('beforeunload', () => {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
        }
    });
});