#%%
import numpy as np
import imageio
import matplotlib.pyplot as plt


file_name = '/data/golam/dnam_wetlab_data/spe/2019_09_13_13_42_03.spe'


image_reader = imageio.get_reader(file_name)
single_image = image_reader.get_data(100)
# 32x32 image crop
x_start = 450
y_start = 450
single_image_cropped = single_image[x_start: x_start + 32, y_start: y_start + 32]

plt.imshow(single_image_cropped, cmap='gray')
# plt.title(f"x_start: {x_start}, y_start: {y_start}")
plt.savefig("experiment_450_450.png", bbox_inches='tight', pad_inches=0)
plt.show()
