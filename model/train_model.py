# scripts/train_model.py

import tensorflow as tf
from tf.keras.models import Sequential
from tf.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Build the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Assuming 3 classes: AM, FM, QPSK
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset_split/train',
    image_size=(128, 128),
    batch_size=32,
    color_mode='grayscale',
    label_mode='int'
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset_split/validation',
    image_size=(128, 128),
    batch_size=32,
    color_mode='grayscale',
    label_mode='int'
)

# Train the model
model.fit(train_dataset, validation_data=validation_dataset, epochs=10)

# Save the model
model.save('models/modulation_recognition_model.h5')
