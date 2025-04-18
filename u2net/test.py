from u2_net import remove_background_single_image
from PIL import Image
import time

start_time = time.time()
image = remove_background_single_image('test_images/cloth2.jpg', model="u2net")
end_time = time.time()

print(f"Time taken: {end_time - start_time} seconds")

# save image to test_images/cloth_u2net.png
image.save('test_images/cloth2_u2net.png')

# save image to test_images/cloth_u2netp.png
image.save('test_images/cloth2_u2netp.png')

print(f"Time taken: {end_time - start_time} seconds")