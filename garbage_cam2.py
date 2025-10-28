import os
import time
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# ---- Load the model ----
MODEL_PATH = "garbage_model2.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded successfully!")
print("Input shape:", input_details[0]['shape'])
print("Output shape:", output_details[0]['shape'])

# ---- Define class labels (according to your model) ----
labels = [
    "cardboard", "glass", "metal", "paper", "plastic",
    "battery", "biological", "clothes", "shoes", "trash"
]

# ---- Define categories ----
biodegradable = {"cardboard", "biological", "clothes", "trash", "paper"}
non_biodegradable = {"glass", "metal", "shoes", "battery", "plastic"}

# ---- Continuous classification loop ----
while True:
    # Capture image from webcam
    os.system("fswebcam -r 640x480 --no-banner /tmp/frame.jpg -S 2 > /dev/null 2>&1")

    # Load and preprocess the image
    img = Image.open("/tmp/frame.jpg").convert("RGB").resize((224, 224))
    input_data = np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

    # Inference
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    end = time.time()

    # Output prediction
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
    class_idx = np.argmax(output_data)
    prob = output_data[class_idx]
    detected_label = labels[class_idx]

    # Determine category
    if detected_label in biodegradable:
        category = "Biodegradable"
    elif detected_label in non_biodegradable:
        category = "Non-Biodegradable"
    else:
        category = "Unknown"

    print(f"Detected: {detected_label} ({prob:.2f}) → {category} | "
          f"Time: {(end-start)*1000:.1f} ms | FPS: {1/(end-start):.2f}")

    # Wait briefly before next capture
    time.sleep(1)
