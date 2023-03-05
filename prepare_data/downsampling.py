# Import the Images module from pillow
from PIL import Image
from dotenv import load_dotenv
import os
load_dotenv()
data_path = os.environ.get('DATA_PATH')

image_path = f"{data_path}test_image.jpeg"

image_file = Image.open(image_path)


# the default
image_file.save(f"{data_path}test_image_downsized_1.jpeg", quality=95)

# Changing the image resolution using quality parameter
# Example-1
image_file.save(f"{data_path}test_image_downsized_2.jpeg", quality=25)

# Example-2
image_file.save(f"{data_path}test_image_downsized_3.jpeg", quality=1)