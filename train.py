import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# ================= CONFIG =================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "training", "train")
VAL_DIR = os.path.join(BASE_DIR, "training", "val")
WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5"
)

# ================= DATA =================
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_data = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

NUM_CLASSES = train_data.num_classes

# ================= MODEL =================
base_model = MobileNetV2(
    weights=WEIGHTS_PATH,
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ================= TRAIN =================
model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# ================= FINE TUNING =================
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# ================= SAVE =================
model.save("crop_disease_model.h5")

class_names = list(train_data.class_indices.keys())
with open("class_names.json", "w") as f:
    json.dump(class_names, f)

print("✅ Training complete")
print("✅ Model saved as crop_disease_model.h5")
print("✅ Classes saved as class_names.json")
