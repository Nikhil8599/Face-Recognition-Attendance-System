const startMarkBtn = document.getElementById("startMarkBtn");
const stopMarkBtn = document.getElementById("stopMarkBtn");
const markVideo = document.getElementById("markVideo");
const markStatus = document.getElementById("markStatus");
const recognizedList = document.getElementById("recognizedList");

let markStream = null;
let markInterval = null;
let recognizedIds = new Set();
async function checkModelBeforeStart() {
    try {
        const res = await fetch('/check_model');
        const data = await res.json();

        if (!data.model_loaded) {
            markStatus.innerHTML = `
                <div class="alert alert-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>Model Not Trained</strong><br>
                    Please train the model first from the Dashboard.
                    <div class="mt-2">
                        <a href="/" class="btn btn-sm btn-primary">Go to Dashboard</a>
                    </div>
                </div>
            `;
            startMarkBtn.disabled = true;
            return false;
        }

        markStatus.innerHTML = `
            <div class="alert alert-success">
                <i class="fas fa-check-circle"></i>
                <strong>Model Ready</strong><br>
                Model trained with ${data.train_samples || 0} students
            </div>
        `;
        return true;
    } catch (error) {
        console.error('Error checking model:', error);
        markStatus.innerHTML = `
            <div class="alert alert-danger">
                <i class="fas fa-times-circle"></i>
                <strong>Error</strong><br>
                Failed to check model status
            </div>
        `;
        return false;
    }
}
document.addEventListener("DOMContentLoaded", checkModelBeforeStart);

startMarkBtn.addEventListener("click", async () => {
    const modelReady = await checkModelBeforeStart();
    if (!modelReady) {
        alert("Cannot start: Model is not trained. Please train the model first.");
        return;
    }

    startMarkBtn.disabled = true;
    stopMarkBtn.disabled = false;
    try {
        markStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 640,
                height: 480,
                facingMode: 'user'
            }
        });
        markVideo.srcObject = markStream;
        await markVideo.play();
        markStatus.innerHTML = '<span class="text-success"><i class="fas fa-circle"></i> Scanning for faces...</span>';
        markInterval = setInterval(captureAndRecognize, 1500);
    } catch (err) {
        alert("Camera error: " + err.message);
        startMarkBtn.disabled = false;
        stopMarkBtn.disabled = true;
    }
});

stopMarkBtn.addEventListener("click", () => {
    if (markInterval) clearInterval(markInterval);
    if (markStream) markStream.getTracks().forEach(t => t.stop());
    startMarkBtn.disabled = false;
    stopMarkBtn.disabled = true;
    markStatus.innerHTML = '<span class="text-muted"><i class="fas fa-circle"></i> Stopped</span>';
});

async function captureAndRecognize() {
    if (!markVideo.videoWidth) return;

    const canvas = document.createElement("canvas");
    canvas.width = markVideo.videoWidth;
    canvas.height = markVideo.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(markVideo, 0, 0, canvas.width, canvas.height);
    markVideo.style.boxShadow = "0 0 0 5px #4CAF50";
    setTimeout(() => {
        markVideo.style.boxShadow = "none";
    }, 300);

    const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.9));
    const fd = new FormData();
    fd.append("image", blob, "snap.jpg");

    try {
        const res = await fetch("/recognize_face", { method: "POST", body: fd });
        const j = await res.json();

        if (j.recognized) {
            markStatus.innerHTML = `
                <span class="text-success">
                    <i class="fas fa-check-circle"></i>
                    Recognized: <strong>${j.name}</strong>
                    (${Math.round(j.confidence * 100)}% confidence)
                    ${j.already_marked ? '<span class="badge bg-warning">Already marked today</span>' : ''}
                </span>
            `;

            if (!recognizedIds.has(j.student_id)) {
                recognizedIds.add(j.student_id);

                const li = document.createElement("li");
                li.className = "list-group-item d-flex justify-content-between align-items-center";
                li.innerHTML = `
                    <div>
                        <strong>${j.name}</strong><br>
                        <small class="text-muted">Class: ${j.class}</small>
                    </div>
                    <div class="text-end">
                        <small>${new Date().toLocaleTimeString()}</small><br>
                        <span class="badge bg-success">${Math.round(j.confidence * 100)}%</span>
                    </div>
                `;


                if (recognizedList.firstChild) {
                    recognizedList.insertBefore(li, recognizedList.firstChild);
                } else {
                    recognizedList.appendChild(li);
                }


                while (recognizedList.children.length > 10) {
                    recognizedList.removeChild(recognizedList.lastChild);
                }


                showNotification(`Attendance marked for ${j.name}`, 'success');
            }
        } else {
            markStatus.innerHTML = `
                <span class="text-warning">
                    <i class="fas fa-exclamation-triangle"></i>
                    ${j.error || 'Not recognized'}
                </span>
            `;
        }
    } catch (err) {
        console.error("Recognition error:", err);
        markStatus.innerHTML = `
            <span class="text-danger">
                <i class="fas fa-times-circle"></i>
                Connection error
            </span>
        `;
    }
}

function showNotification(message, type = 'info') {

    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        min-width: 300px;
    `;
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;

    document.body.appendChild(notification);


    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 3000);
}