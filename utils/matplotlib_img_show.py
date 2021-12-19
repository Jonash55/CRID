import numpy as np
import matplotlib.pyplot as plt


def matplotlib_imshow(img):
    img = img / 2 + 0.5
    npimg = img.cpu().numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
