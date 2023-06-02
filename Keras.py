from tensorflow import keras
import imageio
from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os
os.chdir("/Users/clarasofiechristiansen/Documents/Clara/DTU/Data_Fagprojekt/FagProject")

batch_size = 20
num_channels = 1
num_classes = 4
image_size = 600
latent_dim = 128

submissions = pd.read_csv('submissions.csv')
data = submissions[0:1000]
upvotes = data["Score"].to_numpy()
logupvotes = np.log(upvotes+1)
data["Log_Upvotes"]=logupvotes

quantiles=[0.25,0.5,0.75]
labels=np.quantile(data["Log_Upvotes"],quantiles)


os.chdir("/Users/clarasofiechristiansen/Documents/Clara/DTU/Data_Fagprojekt/FagProject/resized_images/")
X = np.zeros((len(data),600,600))
y=np.zeros(len(data))
for i, row in data.iterrows():
    temp_labels=np.append(labels,row["Log_Upvotes"])
    y[i]=int(np.where(np.sort(temp_labels)==row["Log_Upvotes"])[0][0])
    image = np.asarray(Image.open('EarthPorn-' + str(row["ID"]) + '.png').convert('L'))
    X[i] = image
print(np.count_nonzero(y==0))
print(np.count_nonzero(y==1))
print(np.count_nonzero(y==2))
print(np.count_nonzero(y==3))


X = X.astype("float32") / 255.0
X = np.reshape(X, (-1, 600, 600, 1))
y = keras.utils.to_categorical(y, 4)
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size)

print(f"Shape of training images: {X.shape}")
print(f"Shape of training labels: {y.shape}")


generator_in_channels = latent_dim + num_classes
discriminator_in_channels = num_channels + num_classes
print(generator_in_channels, discriminator_in_channels)

#Create discriminator and generator
generator = keras.Sequential(
    [
        keras.layers.InputLayer((generator_in_channels,)),
        keras.layers.Dense(15 * 15 * generator_in_channels),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Reshape((15, 15, generator_in_channels)),
        keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(1, (7, 7), padding="same", activation="sigmoid"),
        keras.layers.UpSampling2D(size=(10, 10))  # Adjust the upsampling size
    ],
    name="generator",
)

discriminator = keras.Sequential(
    [
        keras.layers.InputLayer((image_size, image_size, discriminator_in_channels)),
        keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.Conv2D(128, (3, 3), strides=(2, 2), padding="same"),
        keras.layers.LeakyReLU(alpha=0.2),
        keras.layers.GlobalMaxPooling2D(),
        keras.layers.Dense(1),
    ],
    name="discriminator",
)

class ConditionalGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super().__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.gen_loss_tracker = keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data.
        real_images, one_hot_labels = data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the images. This is for the discriminator.
        image_one_hot_labels = one_hot_labels[:, :, None, None]
        image_one_hot_labels = tf.repeat(
            image_one_hot_labels, repeats=[image_size * image_size]
        )
        image_one_hot_labels = tf.reshape(
            image_one_hot_labels, (-1, image_size, image_size, num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Decode the noise (guided by labels) to fake images.
        generated_images = self.generator(random_vector_labels)

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_image_and_labels = tf.concat([generated_images, image_one_hot_labels], -1)
        real_image_and_labels = tf.concat([real_images, image_one_hot_labels], -1)
        combined_images = tf.concat(
            [fake_image_and_labels, real_image_and_labels], axis=0
        )

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=1
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_vector_labels)
            fake_image_and_labels = tf.concat([fake_images, image_one_hot_labels], -1)
            predictions = self.discriminator(fake_image_and_labels)
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
        }


"""# Save images
class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=1, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(generator_in_channels,))
        
        for i in range(self.num_img):
            generated_image = generator(random_latent_vectors[i], training=False)
            generated_image = (generated_image * 0.5 + 0.5) * 255  # Rescale values to [0, 255]
            img = keras.preprocessing.image.array_to_img(generated_image[i])
            img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))
cbk = GANMonitor(num_img=3, latent_dim=latent_dim)
"""
cond_gan = ConditionalGAN(
discriminator=discriminator, generator=generator, latent_dim=latent_dim
)
cond_gan.compile(
    d_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0003),
    g_optimizer=keras.optimizers.legacy.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)
n_epochs = 1
cond_gan.fit(dataset, epochs=n_epochs) #,callbacks=[cbk]


# We first extract the trained generator from our Conditional GAN.
trained_gen = cond_gan.generator

# Generate new images using the trained generator
num_samples = 16
latent_vectors = tf.random.normal(shape=(num_samples, latent_dim))
one_hot_labels = tf.one_hot([0, 1, 2, 3], num_classes) 
random_vector_labels = tf.concat([latent_vectors, tf.tile(one_hot_labels, [int(num_samples/num_classes), 1])], axis=1)
generated_images = trained_gen(random_vector_labels, training=False)

# Rescale and plot the generated images
generated_images = (generated_images * 0.5 + 0.5) * 255
fig, axs = plt.subplots(4, 4, figsize=(10, 10))
for i in range(4):
    for j in range(4):
        axs[i][j].imshow(generated_images[4*i+j, :, :, 0], cmap="gray")
        axs[i][j].axis("off")
plt.show()

#Save images
os.chdir("/Users/clarasofiechristiansen/Documents/Clara/DTU/Data_Fagprojekt/FagProject/GAN_keras_examples")
for i in range(num_samples):
    img = generated_images[i, :, :, :].numpy()
    img = keras.utils.array_to_img(img)
    img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=n_epochs))