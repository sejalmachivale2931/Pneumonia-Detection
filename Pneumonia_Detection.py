import tensorflow as tf
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix

train_dir = "C:/Users/Prasad/OneDrive/Desktop/SEJAL/Pneumonia/train"
val_dir = "C:/Users/Prasad/OneDrive/Desktop/SEJAL/Pneumonia/val"

data = []

for label in ["NORMAL", "PNEUMONIA"]:

    path = os.path.join(train_dir, label)

    for img in os.listdir(path):
        data.append([img, label])

df = pd.DataFrame(data, columns=["image", "label"])

print("Dataset Summary")
print(df.head())

print("\nClass Distribution")
print(df["label"].value_counts())

img_size = 150

train_data = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_data = ImageDataGenerator(rescale=1. / 255)

train = train_data.flow_from_directory(

    train_dir,

    target_size=(150, 150),

    batch_size=32,

    class_mode="binary"
)

val = val_data.flow_from_directory(

    val_dir,

    target_size=(150, 150),

    batch_size=32,

    class_mode="binary"
)

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),

    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation="relu"),

    tf.keras.layers.Dense(1, activation="sigmoid")

])

model.compile(

    optimizer="adam",

    loss="binary_crossentropy",

    metrics=["accuracy"]

)

history = model.fit(

    train,

    validation_data=val,

    epochs=5
)

model.save("pneumonia_model.keras")

print("Training Complete")

img = image.load_img(

    "C:/Users/Prasad/OneDrive/Desktop/SEJAL/Pneumonia/New.jpg",

    target_size=(150, 150)
)

img_array = image.img_to_array(img)

img_array = img_array / 255

img_array = np.expand_dims(img_array, axis=0)

result = model.predict(img_array)

print("Prediction Probability:", result)

if result[0][0] > 0.5:

    print("Prediction: Pneumonia")

else:

    print("Prediction: Normal")

pred = model.predict(val)

y_pred = (pred > 0.5)

print(confusion_matrix(val.classes,y_pred))

print(classification_report(val.classes,y_pred))

plt.plot(history.history["accuracy"])

plt.plot(history.history["val_accuracy"])

plt.title("Model Accuracy")

plt.ylabel("Accuracy")

plt.xlabel("Epoch")

plt.legend(["Train","Validation"])

plt.show()