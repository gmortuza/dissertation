import numpy as np
import matplotlib.pyplot as plt
from io_modified import load_raw

images = load_raw("simulation.raw")[0]
plt.imshow(images[3], cmap="gray")
plt.show()