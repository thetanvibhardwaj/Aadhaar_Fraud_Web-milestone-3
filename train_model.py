import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

data_dir = "dataset"

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train = datagen.flow_from_directory(
    data_dir,
    target_size=(224,224),
    batch_size=8,
    class_mode='binary',
    subset='training'
)

val = datagen.flow_from_directory(
    data_dir,
    target_size=(224,224),
    batch_size=8,
    class_mode='binary',
    subset='validation'
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train, epochs=5, validation_data=val)

model.save("fraud_model.h5")

print("Model trained successfully!")