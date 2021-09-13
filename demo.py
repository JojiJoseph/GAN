import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tensorflow as tf

from gan import Generator

gen = Generator()
gen.build([1, 100])
gen.load_weights("./gen.h5")

st.title("GAN Demo")
fig = plt.figure()
for i in range(9):
    plt.subplot(3, 3, i+1)
    z = np.random.normal(size=(1, 100))
    y = gen.predict(z)
    y = tf.reshape(y, (28, 28, 1))
    plt.imshow(y, cmap="gray")

st.pyplot(fig)
