# Import the Images module from pillow
from PIL import Image

# Open the image by specifying the image path.
image_path = "../data/test_image.jpg"
image_file = Image.open(image_path)

# the default
image_file.save("data/test_image_downsized_1.jpg", quality=95)

# Changing the image resolution using quality parameter
# Example-1
image_file.save("data/test_image_downsized_2.jpg", quality=25)

# Example-2
image_file.save("data/test_image_downsized_3.jpg", quality=1)