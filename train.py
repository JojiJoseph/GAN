import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
from tqdm import tqdm

from gan import Disc, Generator

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tfk.datasets.mnist.load_data()

    x_train = x_train.reshape((-1, 784)).astype('float')/255.

    batch_size = 100
    total_len = x_train.shape[0]

    gen = Generator()
    disc = Disc()

    opt_gen = tfk.optimizers.Adam(learning_rate=0.0001)
    opt_disc = tfk.optimizers.Adam(learning_rate=0.0001)

    bce = tfk.losses.BinaryCrossentropy()
    for epoch in range(1_000):
        for batch_id in tqdm(range(total_len // batch_size)):

            # Train generator
            with tf.GradientTape() as tape:
                z = np.random.normal(size=(batch_size, 100))
                y_g = gen(z)
                y_d = disc(y_g)
                loss = bce(np.ones((batch_size, 1)), y_d)
                grads = tape.gradient(loss, gen.trainable_variables)
            opt_gen.apply_gradients(zip(grads, gen.trainable_variables))

            # Train discriminator
            with tf.GradientTape() as tape:
                y_d = disc(y_g)
                loss = bce(np.zeros((batch_size, 1)), y_d)
                y_d = disc(
                    x_train[batch_id*batch_size: (batch_id+1)*batch_size])
                loss += bce(np.ones((batch_size, 1)), y_d)
                grads = tape.gradient(loss, disc.trainable_variables)
            opt_disc.apply_gradients(zip(grads, disc.trainable_variables))
        disc.save_weights("./disc.h5")
        gen.save_weights("./gen.h5")

        for i in range(9):
            plt.subplot(3, 3, i+1)
            z = np.random.normal(size=(1, 100))
            y = gen(z)
            y = tf.reshape(y, (28, 28, 1))
            plt.imshow(y, cmap="gray")
        plt.savefig(f"./fig{epoch}.png")
    plt.show()
