import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import tensorflow.keras as tfk

from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from cgan import Disc, Generator

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = tfk.datasets.mnist.load_data()

    x_train = x_train.reshape((-1, 784)).astype('float')/255.

    batch_size = 100
    total_len = x_train.shape[0]

    gen = Generator().to(device)
    disc = Disc().to(device)

    opt_gen = torch.optim.Adam(gen.parameters(), lr=1e-4)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=1e-4)

    bce = torch.nn.BCELoss()
    for epoch in range(1_000):
        for batch_id in tqdm(range(total_len // batch_size)):
            label_batch = y_train[batch_id*batch_size: (batch_id+1)*batch_size]
            label_batch = torch.from_numpy(label_batch).long().to(device)
            real_batch = x_train[batch_id*batch_size: (batch_id+1)*batch_size]
            real_batch = torch.from_numpy(real_batch).float().to(device)
            # Train generator
            # with tf.GradientTape() as tape:
            z = np.random.normal(size=(batch_size, 100))
            z = torch.from_numpy(z).float().to(device)
            y_g = gen(z,label_batch)
            y_d = disc(y_g,label_batch)
            loss = bce(y_d, torch.ones((batch_size, 1)).to(device))
            # grads = tape.gradient(loss, gen.trainable_variables)
            opt_gen.zero_grad()
            loss.backward()
            opt_gen.step()
            # opt_gen.apply_gradients(zip(grads, gen.trainable_variables))

            # Train discriminator
            # with tf.GradientTape() as tape:
            y_g = y_g.detach()
            y_d = disc(y_g,label_batch)
            loss = bce(y_d, torch.zeros((batch_size, 1)).to(device))
            y_d = disc(
                real_batch, label_batch)
            loss += bce(y_d, torch.ones((batch_size, 1)).to(device))
            opt_disc.zero_grad()
            loss.backward()
            # grads = tape.gradient(loss, disc.trainable_variables)
            opt_disc.step()#apply_gradients(zip(grads, disc.trainable_variables))
        # disc.save_weights("./disc_cgan.h5")
        # gen.save_weights("./gen_cgan.h5")
        torch.save(disc.state_dict(), "./disc_cgan.pt")
        torch.save(gen.state_dict(), "./gen_cgan.pt")

        for i in range(10):
            plt.subplot(2, 5, i+1)
            z = np.random.normal(size=(1, 100))
            z = torch.from_numpy(z).float().to(device)
            cat = np.array([i])
            cat = torch.from_numpy(cat).long().to(device)
            gen.eval()
            with torch.no_grad():
                y = gen(z, cat)
            y = y.view(28,28,1)
            # y = torch.reshape(y, (28, 28, 1))
            gen.train()
            plt.imshow(y.cpu().numpy(), cmap="gray")
        plt.savefig(f"./cgan_fig{epoch}.png")
    plt.show()
