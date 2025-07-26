import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report

# Paths to the dataset directories
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# Image dimensions
img_width, img_height = 150, 150

# Set up ImageDataGenerator for training with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

# Validation DataGenerator (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)

# Load images in batches from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=8,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=8,
    class_mode='binary',
    shuffle=False  # Important for confusion matrix to align labels and predictions
)

# Print class mapping to verify correct labeling
print("Class mapping:", train_generator.class_indices)

# Define the CNN model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model with Adam optimizer and binary crossentropy loss
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('deepfake_cnn_model.keras', monitor='val_loss', save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    callbacks=[early_stop, checkpoint, reduce_lr]
)

# Save the final model (although checkpoint saves the best one)
model.save('deepfake_cnn_model.keras')
print("✅ Model training complete and saved as 'deepfake_cnn_model.keras'.")

# Evaluate on validation data
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"📊 Validation accuracy: {val_accuracy * 100:.2f}%")

# Predict on validation data
y_true = val_generator.classes
y_pred_prob = model.predict(val_generator, steps=val_generator.samples // val_generator.batch_size + 1)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)[:len(y_true)]  # trim any extra predictions

# Confusion Matrix and Classification Report
cm = confusion_matrix(y_true, y_pred)
print("\n🔍 Confusion Matrix:")
print(cm)
print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred, target_names=['real', 'fake']))

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['real', 'fake'], yticklabels=['real', 'fake'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
