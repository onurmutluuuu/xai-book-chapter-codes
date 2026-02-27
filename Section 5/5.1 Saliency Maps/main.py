import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import matplotlib.pyplot as plt

# 1. Preparation of Model and Data
# Load the VGG16 model pre-trained with ImageNet weights.
vgg_model = VGG16(weights='imagenet')

# Instead of a real image, create a random tensor representing 
# 'noise' or a sample data point for analysis.
img_tensor = tf.random.uniform((1, 224, 224, 3), minval=0, maxval=255)
img_processed = preprocess_input(tf.identity(img_tensor))

print("\n--- Initiating Saliency Map Analysis ---")

# 2. Gradient Calculation Process (Sensitivity Analysis)
# Compute the derivative of the class score with respect to the input using GradientTape
with tf.GradientTape() as tape:
    tape.watch(img_processed)
    preds = vgg_model(img_processed)
    top_pred_index = tf.argmax(preds[0])
    top_class_score = preds[0, top_pred_index]

# Gradient: w = ∂Sc / ∂x (Derivative calculation)
gradients = tape.gradient(top_class_score, img_processed)

# 3. Preparation for Visualization
# Take the absolute value of the gradients and find the maximum across RGB channels.
saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1)[0]

# Normalization: Rescale to 0-1 range to enhance visual quality.
saliency_min = tf.reduce_min(saliency_map)
saliency_max = tf.reduce_max(saliency_map)
saliency_map = (saliency_map - saliency_min) / (saliency_max - saliency_min)

# 4. Visualization of Results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Representation of the Original Image
# (Visualized by reverting preprocess_input effects)
axes[0].imshow(img_tensor[0].numpy().astype(np.uint8))
axes[0].set_title("Input Image (Observation Field)")
axes[0].axis('off')

# Saliency Map
im = axes[1].imshow(saliency_map, cmap='hot')
axes[1].set_title("Saliency Map (Sensitivity Map)")
axes[1].axis('off')

plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
plt.show()

print("Description: The heatmap highlights the pixels (bright areas) that most significantly influence the model's prediction.")