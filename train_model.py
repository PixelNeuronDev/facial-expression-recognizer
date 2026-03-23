import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam

# 1. Setup Paths
# 1. Setup Paths (Using 'r' for raw strings and fixing the folders)
train_dir = r'C:\Users\k3796\OneDrive\Desktop\Facial_Expression\archive\train'
test_dir  = r'C:\Users\k3796\OneDrive\Desktop\Facial_Expression\archive\test'

# 2. Data Augmentation (The "Anti-Rubbish" Shield)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, color_mode='grayscale', target_size=(48, 48),
    batch_size=32, class_mode='categorical', shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    test_dir, color_mode='grayscale', target_size=(48, 48),
    batch_size=32, class_mode='categorical', shuffle=True)

# 3. The Model Architecture
model = Sequential([
    # Block 1
    Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(48,48,1)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (5,5), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(7, activation='softmax')
])
# Print the class indices so we can update our camera script later
print("Class Indices (The Dictionary):")
print(train_generator.class_indices)
# 4. Compile and Train
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting Training... This might take a while!")
model.fit(train_generator, epochs=50, validation_data=validation_generator)

# 5. Save the Brain
model.save('emotion_model.h5')
print("Model Saved as emotion_model.h5")