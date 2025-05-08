import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load images
img1 = mpimg.imread('testing_shape.png')
img2 = mpimg.imread('snapshot01.png')
# img3 = mpimg.imread('curvature_only_one.png')

# Create figure
fig, axs = plt.subplots(1, 2, figsize=(12, 4))  # 1 row, 3 columns

axs[0].imshow(img1)
axs[1].imshow(img2)
# axs[2].imshow(img3)

# Remove axes and add titles if needed
for ax in axs:
    ax.axis('off')  # remove axes
    # ax.set_title('Optional Title')

plt.tight_layout()
plt.savefig('combined_image.png', dpi=300)
plt.show()