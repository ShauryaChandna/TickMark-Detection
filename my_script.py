import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

def load_dataset(positive_dir, negative_dir, img_size=(224, 224)):  
    data = []
    labels = []

    for img_name in os.listdir(positive_dir):
        img_path = os.path.join(positive_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size) / 255.0
            data.append(img)
            labels.append(1)

    for img_name in os.listdir(negative_dir):
        img_path = os.path.join(negative_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size) / 255.0
            data.append(img)
            labels.append(0)

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    return data, labels

positive_dir = "/Users/shauryachandna/Documents/Positive"
negative_dir = "/Users/shauryachandna/Documents/Negative"
img_size = (224, 224) 

data, labels = load_dataset(positive_dir, negative_dir, img_size)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint(
    filepath="mobilenetv2_tick_mark_model.keras",
    monitor="val_loss",
    save_best_only=True,
    mode="auto",
    verbose=1
)

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=16,
    callbacks=[checkpoint]
)

test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

model.save("mobilenetv2_tick_mark_model_final.keras")
print("Final model saved as mobilenetv2_tick_mark_model_final.keras")
