import numpy as np
import matplotlib.pyplot as plt
from io_modified import load_raw
import pandas as pd
import hd5

images = load_raw("simulation.raw")[0]
plt.imshow(images[5], cmap="gray")
plt.show()