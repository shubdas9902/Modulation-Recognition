# evaluate_model.py

import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model('models/modulation_recognition_model.h5')

# Load test dataset
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'dataset_split/test',
    image_size=(128, 128),
    batch_size=32,
    color_mode='grayscale',
    label_mode='int'
)

# Evaluate the model
loss, accuracy = model.evaluate(test_dataset)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# Predictions
y_true = []
y_pred = []
for images, labels in test_dataset:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(predictions.argmax(axis=1))

# Classification report
print(classification_report(y_true, y_pred))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
