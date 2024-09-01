# visualize_predictions.py

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('models/modulation_recognition_model.h5')

# Load test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset_split/test',
    image_size=(128, 128),
    batch_size=1,
    color_mode='grayscale',
    label_mode='int'
)

# Predict and visualize
for images, labels in test_dataset.take(5):  # Change the number to visualize more samples
    predictions = model.predict(images)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(images[0].numpy().reshape(128, 128), cmap='gray')
    plt.title('Input Image')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(3), predictions[0])
    plt.title('Prediction Scores')
    plt.xticks(range(3), ['AM', 'FM', 'QPSK'])
    plt.show()
