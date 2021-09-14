import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import torch

from cgan import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gen = Generator().to(device)
gen.load_state_dict(torch.load("./gen_cgan.pt"))
gen.eval()

st.title("Conditional GAN Demo")
use_zero_noise = st.checkbox("Use zero noise", True)
fig = plt.figure()
for i in range(10):
    plt.subplot(2, 5, i+1)
    z = np.random.normal(size=(1, 100))
    z = torch.from_numpy(z).float().to(device)
    if use_zero_noise:
        z = torch.zeros((1,100)).float().to(device)
    with torch.no_grad():
        cat = torch.from_numpy(np.array([i])).long().to(device)
        y = gen(z, cat)
        y = y.view(28,28,1)
    plt.axis("off")
    plt.imshow(y.cpu().numpy(), cmap="gray")

st.pyplot(fig)
