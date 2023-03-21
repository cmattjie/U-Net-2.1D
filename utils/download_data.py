import gdown

# Set the URL of the folder you want to download
url = "https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2"

# Set the output directory where you want to save the downloaded folder
output_dir = "/mnt/B-SSD/unet21d_slices/datasets/drive"

# Download the folder using gdown
gdown.download(url, output_dir, quiet=False)
