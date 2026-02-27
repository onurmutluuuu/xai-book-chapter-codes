
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from tensorflow.keras.models import Model
import cv2
import requests
from io import BytesIO

# --- 1. DATA LOADING AND PREPARATION ---
# Store original dimensions to maintain image clarity during visualization
url = "https://images.unsplash.com/photo-1500514966906-fe245eea9344?q=80&w=600&auto=format&fit=crop"
headers = {'User-Agent': 'Mozilla/5.0'}

try:
    response = requests.get(url, headers=headers)
    # Load the original image in high resolution (for visualization)
    img_pil = image.load_img(BytesIO(response.content)) 
    orig_w, orig_h = img_pil.size
    
    # Low-resolution copy required for the model input
    img_model = img_pil.resize((224, 224))
    img_array_model = image.img_to_array(img_model)
    img_array_display = image.img_to_array(img_pil) # Original array for sharp visualization
    
    print(f"Image loaded successfully. Original Dimensions: {orig_w}x{orig_h}")
except Exception as e:
    print(f"Error: {e}")

# Preprocessing for the model (based on 224x224 input)
img_tensor = np.expand_dims(img_array_model, axis=0)
img_tensor = preprocess_input(img_tensor)

# --- 2. GRAD-CAM CALCULATION ---
base_model = VGG16(weights='imagenet', include_top=True)
last_conv_layer = base_model.get_layer("block5_conv3")
grad_model = Model([base_model.input], [last_conv_layer.output, base_model.output])

with tf.GradientTape() as tape:
    conv_outputs, predictions = grad_model(img_tensor)
    top_class_idx = tf.argmax(predictions[0])
    loss = predictions[:, top_class_idx]

# Gradients and channel weights
grads = tape.gradient(loss, conv_outputs)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# Heatmap generation
last_conv_output = conv_outputs[0]
heatmap = last_conv_output @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)
heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
heatmap = heatmap.numpy()

# --- 3. HIGH-RESOLUTION VISUALIZATION ---
# 1. Resize the heatmap to the original image dimensions (Preserving clarity)
heatmap_resized = cv2.resize(heatmap, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
heatmap_resized = np.uint8(255 * heatmap_resized)
heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

# 2. Superimpose with the original high-resolution image
superimposed_img = heatmap_color * 0.5 + img_array_display
superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

# 3. Plotting (Output with high DPI)
plt.rcParams['figure.dpi'] = 140 # Increases screen resolution
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].imshow(img_array_display.astype(np.uint8))
axes[0].set_title(f"Original Image ({orig_w}x{orig_h})", fontsize=12)
axes[0].axis('off')

axes[1].imshow(superimposed_img)
axes[1].set_title("Grad-CAM: High-Resolution Heatmap", fontsize=12)
axes[1].axis('off')

plt.tight_layout()
plt.show()

# Prediction result
decoded = decode_predictions(predictions.numpy(), top=1)[0][0]
print(f"Model Prediction: {decoded[1]} (Confidence Score: %{decoded[2]*100:.2f})")