import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, Normalize, PILToTensor
from torchvision.transforms.transforms import ConvertImageDtype

from cgan import Disc, Generator
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dataset = torchvision.datasets.MNIST("./data", download=True, transform=Compose(
    [PILToTensor(), ConvertImageDtype(torch.float), Normalize(0, 1), ]))

batch_size = 100
data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=10)

if __name__ == "__main__":

    gen = Generator().to(device)
    disc = Disc().to(device)

    opt_gen = torch.optim.Adam(gen.parameters(), lr=1e-4)
    opt_disc = torch.optim.Adam(disc.parameters(), lr=1e-4)

    bce = torch.nn.BCELoss()
    for epoch in range(1_000):
        print(f"Epoch {epoch+1} ...")
        for batch_id, (real_batch, label_batch) in tqdm(enumerate(data_loader)):

            label_batch = label_batch.to(device)
            real_batch = torch.permute(real_batch, [0, 2, 3, 1])
            real_batch = real_batch.view(-1, 784).to(device)
            
            # Train generator
            z = np.random.normal(size=(batch_size, 100))
            z = torch.from_numpy(z).float().to(device)
            y_g = gen(z, label_batch)
            y_d = disc(y_g, label_batch)
            loss = bce(y_d, torch.ones((batch_size, 1)).to(device))
            opt_gen.zero_grad()
            loss.backward()
            opt_gen.step()

            # Train discriminator
            y_g = y_g.detach()
            y_d = disc(y_g, label_batch)
            loss = bce(y_d, torch.zeros((batch_size, 1)).to(device))
            y_d = disc(
                real_batch, label_batch)
            loss += bce(y_d, torch.ones((batch_size, 1)).to(device))
            opt_disc.zero_grad()
            loss.backward()
            opt_disc.step()  # apply_gradients(zip(grads, disc.trainable_variables))
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
            y = y.reshape(28, 28, 1)
            gen.train()
            plt.imshow(y.cpu().numpy(), cmap="gray")
        plt.savefig(f"./cgan_fig{epoch}.png")
    plt.show()
