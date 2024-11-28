import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.callbacks import Callback

# Define Paths
HR_PATH = '/content/drive/MyDrive/DIV2K_train_HR'  # Path to High-Resolution images
OUTPUT_DIR = './generated_images_3/'  # Output directory for generated images
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('./saved_models/', exist_ok=True)  # Directory to save generator models

# Data Preprocessing
def load_hr_images(path, target_size=(256, 256)):
    images = []
    for img_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, img_name))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            images.append(img / 255.0)  # Normalize to [0, 1]
    return np.array(images, dtype=np.float32)

def generate_lr_images(hr_images, scale=4):
    # Downscale HR images
    lr_images = tf.image.resize(
        hr_images,
        [hr_images.shape[1] // scale, hr_images.shape[2] // scale],
        method=tf.image.ResizeMethod.BICUBIC
    )

    # Upscale back to original size
    lr_images = tf.image.resize(
        lr_images,
        [hr_images.shape[1], hr_images.shape[2]],
        method=tf.image.ResizeMethod.BICUBIC
    )
    return lr_images

# Load HR images
hr_images = load_hr_images(HR_PATH)
hr_images = tf.convert_to_tensor(hr_images, dtype=tf.float32)  # Convert to TensorFlow tensor
lr_images = generate_lr_images(hr_images)

# Define Generator
def build_generator():
    inputs = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(inputs)
    x = layers.PReLU()(x)

    for _ in range(8):
        skip = x
        x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.PReLU()(x)
        x = layers.Add()([skip, x])

    x = layers.Conv2D(3, kernel_size=3, strides=1, padding='same')(x)
    outputs = layers.Activation('tanh')(x)
    return Model(inputs, outputs, name='Generator')

# Define Discriminator
def build_discriminator():
    inputs = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding='same')(inputs)
    x = layers.LeakyReLU()(x)

    for filters in [128, 256, 512]:
        x = layers.Conv2D(filters, kernel_size=3, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs, name='Discriminator')

# Initialize VGG for Perceptual Loss
vgg = VGG19(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
vgg.trainable = False
feature_extractor = Model(inputs=vgg.input, outputs=vgg.get_layer("block5_conv4").output)

# Define Perceptual Loss Function
def perceptual_loss(hr_images, fake_hr_images):
    hr_features = feature_extractor(hr_images)
    fake_hr_features = feature_extractor(fake_hr_images)
    return tf.reduce_mean(tf.square(hr_features - fake_hr_features))

# CGAN Class
class CGAN(tf.keras.Model):
    def __init__(self, generator, discriminator):
        super(CGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.bce = tf.keras.losses.BinaryCrossentropy()

    def compile(self, g_optimizer, d_optimizer):
        super(CGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer

    def train_step(self, data):
        lr_images, hr_images = data  # Unpack LR and HR images from the tuple

        # Train Discriminator
        with tf.GradientTape() as d_tape:
            fake_hr = self.generator(lr_images, training=True)
            real_pred = self.discriminator(hr_images, training=True)
            fake_pred = self.discriminator(fake_hr, training=True)
            d_loss_real = self.bce(tf.ones_like(real_pred), real_pred)
            d_loss_fake = self.bce(tf.zeros_like(fake_pred), fake_pred)
            d_loss = d_loss_real + d_loss_fake

        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Train Generator
        with tf.GradientTape() as g_tape:
            fake_hr = self.generator(lr_images, training=True)
            fake_pred = self.discriminator(fake_hr, training=True)
            g_loss_adv = self.bce(tf.ones_like(fake_pred), fake_pred)
            g_loss_content = perceptual_loss(hr_images, fake_hr)
            g_loss = g_loss_adv + g_loss_content

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))

        return {"d_loss": d_loss, "g_loss": g_loss}

# Callback for Saving Generated Images
class SaveGeneratedImages(tf.keras.callbacks.Callback):
    def __init__(self, generator, output_dir, test_lr_images, save_every=5):
        self.generator = generator
        self.output_dir = output_dir
        self.test_lr_images = test_lr_images
        self.save_every = save_every

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_every == 0:
            generated_images = self.generator.predict(self.test_lr_images)
            for i, img in enumerate(generated_images):
                plt.imsave(f"{self.output_dir}/epoch_{epoch+1}_img_{i+1}.png", (img * 255).astype(np.uint8))

# Callback for Saving Generator Periodically
class SaveGeneratorCallback(tf.keras.callbacks.Callback):
    def __init__(self, generator, save_dir, save_every=10):
        self.generator = generator
        self.save_dir = save_dir
        self.save_every = save_every
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_every == 0:
            save_path = os.path.join(self.save_dir, f'generator_epoch_{epoch+1}.weights.h5')
            self.generator.save_weights(save_path)
            print(f'Generator weights saved at {save_path}')

# Instantiate Models
generator = build_generator()
discriminator = build_discriminator()
cgan = CGAN(generator, discriminator)
cgan.compile(
    g_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5),
    d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5)
)

# Prepare Test Data for Image Saving
test_lr_images = generate_lr_images(tf.convert_to_tensor(hr_images[:5]))  # Use first 5 HR images for visualization
save_callback = SaveGeneratedImages(generator, OUTPUT_DIR, test_lr_images, save_every=5)
save_generator_callback = SaveGeneratorCallback(generator, './saved_models/', save_every=10)

# Combine LR and HR images into a dataset
dataset = tf.data.Dataset.from_tensor_slices((lr_images, hr_images)).batch(16)

# Train the CGAN
cgan.fit(dataset, epochs=200, callbacks=[save_callback, save_generator_callback])
