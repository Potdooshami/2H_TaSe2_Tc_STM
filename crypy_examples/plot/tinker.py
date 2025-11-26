import matplotlib.pyplot as plt
import matplotlib.image as mpimg


filepath = 'assets\dwn_toon.png'
img = mpimg.imread(filepath)
plt.imshow(img)
plt.axis('off')  # Hide axes ticks and labels
plt.show()
