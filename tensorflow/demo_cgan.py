import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

from cgan import Generator

gen = Generator()
gen(np.zeros([1, 100]),np.zeros([1],dtype="uint8"))
gen.load_weights("./gen_cgan.h5")

st.title("Conditional GAN Demo")
use_zero_noise = st.checkbox("Use zero noise", True)
fig = plt.figure()
for i in range(10):
    plt.subplot(2, 5, i+1)
    z = np.random.normal(size=(1, 100))
    if use_zero_noise:
        z = np.zeros((1,100))
    y = gen(z, np.array([i]))
    y = tf.reshape(y, (28, 28, 1))
    plt.axis("off")
    plt.imshow(y, cmap="gray")

st.pyplot(fig)
