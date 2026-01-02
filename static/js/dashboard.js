async function checkModelStatus() {
    try {
        const res = await fetch('/check_model');
        const data = await res.json();

        console.log("Model status:", data);

        if (data.model_loaded) {
            document.getElementById('modelStatus').textContent = 'Active';
            document.getElementById('modelStatus').className = 'status-indicator active';
            document.getElementById('trainingProgress').style.display = 'none';
            document.getElementById('trainingComplete').style.display = 'block';
            document.getElementById('trainingIdle').style.display = 'none';
            document.getElementById('trainStatusText').textContent = 'Model trained and ready';
        } else {
            document.getElementById('modelStatus').textContent = 'Not Trained';
            document.getElementById('modelStatus').className = 'status-indicator inactive';
        }

        return data;
    } catch (error) {
        console.error('Error checking model status:', error);
        return null;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    checkModelStatus();
    setInterval(checkModelStatus, 10000);
});