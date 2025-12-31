from PIL import Image
import os

# Load and resize the large image
img = Image.open('results/visualizations/residuals_grid.png')
print(f'Original size: {img.size}')
print(f'File size: {os.path.getsize("results/visualizations/residuals_grid.png") / 1024 / 1024:.1f} MB')

# Resize to 25% for GitHub
new_size = (img.width // 4, img.height // 4)
img_small = img.resize(new_size, Image.LANCZOS)
img_small.save('results/visualizations/residuals_grid_preview.png', optimize=True)
print(f'New size: {img_small.size}')
print(f'New file size: {os.path.getsize("results/visualizations/residuals_grid_preview.png") / 1024 / 1024:.1f} MB')

