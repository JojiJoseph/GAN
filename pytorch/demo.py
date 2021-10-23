import matplotlib.pyplot as plt
import numpy as np
import torch
import streamlit as st

from gan import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator().to(device)
gen.load_state_dict(torch.load("./gen_gan.pt"))
gen.eval()

st.title("GAN Demo")

fig = plt.figure()

for i in range(9):
    plt.subplot(3, 3, i+1)
    z = np.random.normal(size=(1, 100))
    z = torch.from_numpy(z).float().to(device)
    y = gen(z)
    y = torch.reshape(y, (28, 28, 1))
    y = y.detach().cpu().numpy()
    plt.axis("off")
    plt.imshow(y, cmap="gray")

st.pyplot(fig)
