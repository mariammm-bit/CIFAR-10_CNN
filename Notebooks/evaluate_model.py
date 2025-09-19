import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
import os, random

# ---------- Parameters ----------
img_size = (224, 224)
batch_size = 32

# ---------- Load datasets (من غير normalization هنا) ----------
train_dataset = tf.keras.utils.image_dataset_from_directory(
    r"D:\CIFAR-10_CNN\Processed_train",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    r"D:\CIFAR-10_CNN\Processed_train",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    r"D:\CIFAR-10_CNN\Processed_test",
    image_size=img_size,
    batch_size=batch_size
)

# ---------- Classes ----------
class_names = train_dataset.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# ---------- Data Augmentation ----------
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomZoom(0.1),
])

# ---------- Transfer Learning (MobileNetV2) ----------
base_model = keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze base model

model = Sequential([
    data_augmentation,
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ---------- Training ----------
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=10
)

# ---------- Evaluate ----------
loss, acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {acc*100:.2f}%")

# ---------- Predict random image from test ----------
test_folder = r"D:\CIFAR-10_CNN\Processed_test"
img_name = random.choice(os.listdir(os.path.join(test_folder, class_names[0])))  # صورة عشوائية من أول class
img_path = os.path.join(test_folder, class_names[0], img_name)

print("Testing on:", img_path)
img = tf.keras.utils.load_img(img_path, target_size=img_size)
img_array = tf.keras.utils.img_to_array(img) / 255.0   # لو الـ normalization معمول قبل كدا خليها من غير /255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
pred_class = np.argmax(prediction)

print("Predicted class:", class_names[pred_class])
print("Prediction confidence:", prediction[0][pred_class])
print("All class probabilities:", prediction[0])

# ---------- Save model ----------
model.save(r"D:\CIFAR-10_CNN\CNN_model.h5")
