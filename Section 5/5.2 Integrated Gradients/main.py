import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# --- Setup: Model and Data Preparation ---
vgg_model = VGG16(weights='imagenet')
# Create a sample random tensor (224x224x3) and preprocess it for the VGG16 format
img_tensor_processed = preprocess_input(tf.random.uniform((1, 224, 224, 3), minval=0, maxval=255))

def get_gradients(input_image):
    with tf.GradientTape() as tape:
        tape.watch(input_image)
        preds = vgg_model(input_image)
        # Use index [0] to convert the argmax result into a scalar
        top_pred_index = tf.argmax(preds[0]) 
        top_class_score = preds[0, top_pred_index]
    return tape.gradient(top_class_score, input_image)

print("\n--- Integrated Gradients (IG) ---")
# 1. Establish a baseline (e.g., a black image)
baseline = tf.zeros_like(img_tensor_processed)

# 2. Define integration steps (m) (e.g., 50 steps)
m_steps = 50
alphas = tf.linspace(start=0.0, stop=1.0, num=m_steps)

# 3. Generate the path (interpolation)
interpolated_images = [baseline + alpha * (img_tensor_processed - baseline) for alpha in alphas]
interpolated_images = tf.stack(interpolated_images)

# 4. Compute gradients for each interpolated image
all_gradients = [] # Initialize an empty list
for img in interpolated_images:
    # Expand dimensions as get_gradients expects a 4D (batch) tensor
    all_gradients.append(get_gradients(img))
all_gradients = tf.stack(all_gradients)

# 5. Average the gradients (approximating the integral)
avg_gradients = tf.reduce_mean(all_gradients, axis=0)

# 6. Calculate IG: (input - baseline) * average_gradient
integrated_gradients = (img_tensor_processed - baseline) * avg_gradients

# Sum absolute values across channels and select the first index for visualization
ig_map = tf.reduce_sum(tf.abs(integrated_gradients), axis=-1)[0]
ig_map_normalized = (ig_map - tf.reduce_min(ig_map)) / (tf.reduce_max(ig_map) - tf.reduce_min(ig_map))

# Visualization
plt.figure(figsize=(8, 8))
plt.imshow(ig_map_normalized, cmap='hot')
plt.title("Integrated Gradients (IG) Map")
plt.axis('off')
plt.show()

print("Description: IG addresses the saturation problem and provides a clearer, less noisy feature importance map compared to Saliency Maps.")