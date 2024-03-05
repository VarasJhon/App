// Select elements from the DOM
const video = document.getElementById('video');
const toggleCameraButton = document.getElementById('toggle-camera-btn');
const captureButton = document.getElementById('capture-btn');
const resultDiv = document.getElementById('result');
let stream = null;
let model;

// Load CIFAR-10 model
async function loadModel() {
    model = await tf.loadLayersModel('url_del_modelo/model.json');
}

// Start or stop the camera
async function toggleCamera() {
    if (!stream) {
        try {
            stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' } });
            video.srcObject = stream;
            toggleCameraButton.innerText = 'Desactivar Cámara';
        } catch (err) {
            console.error('Error al acceder a la cámara:', err);
        }
    } else {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
        toggleCameraButton.innerText = 'Activar Cámara';
        video.srcObject = null;
    }
}

// Capture image from the camera
function captureImage() {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
    classifyImage(canvas);
}

// Classify captured image
async function classifyImage(canvas) {
    const img = tf.browser.fromPixels(canvas).resizeBilinear([32, 32]).toFloat().expandDims();
    const predictions = await model.predict(img).data();
    const maxPrediction = Math.max(...predictions);
    const predictedClass = predictions.indexOf(maxPrediction);
    const classNames = ["avión", "automóvil", "pájaro", "gato", "ciervo", "perro", "rana", "caballo", "barco", "camión"];
    resultDiv.innerText = `Clase predicha: ${classNames[predictedClass]}`;
}

// Add click event to the toggle-camera-btn button
toggleCameraButton.addEventListener('click', toggleCamera);
captureButton.addEventListener('click', captureImage);

// Initialize the application
loadModel();
