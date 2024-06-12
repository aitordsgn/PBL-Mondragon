import os
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug import parameters as iap


augmenters = [
    iaa.Crop(percent=(0, 0.1)),  # Random cuts
    iaa.GaussianBlur(sigma=(0, 3.0)),  # Change focus with sigma from 0 to 3.0
    iaa.LinearContrast((0.75, 1.5)),  # Increase or decrease contrast
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # Add Gaussian noise
    iaa.Multiply((0.5, 1.5), per_channel=0.1),  # Lighter or darker images
    iaa.Invert(0.05, per_channel=True),  # Invert color channels
    iaa.Add((-10, 10), per_channel=0.5),  # Add value from -10 a 10 to each píxel
    iaa.Grayscale(alpha=(0.0, 1.0)),  # Convert to grayscale
    iaa.GaussianBlur(sigma=iap.Uniform(0.0, 1.0)),  # Gaussian blur with random sigma
    iaa.Affine(
        rotate=iap.Normal(0.0, 30),
        translate_px=iap.RandomSign(iap.Poisson(3))
    ),
    iaa.AddElementwise(iap.Discretize((iap.Beta(0.5, 0.5) * 2 - 1.0) * 64)),
    iaa.BlendAlpha(
        (0.0, 1.0),
        foreground=iaa.MedianBlur(11),
        per_channel=True
    ),
    iaa.BlendAlphaFrequencyNoise(
        foreground=iaa.Affine(
            rotate=(-10, 10),
            translate_px={"x": (-4, 4), "y": (-4, 4)}
        ),
        background=iaa.AddToHueAndSaturation((-40, 40)),
        per_channel=0.5
    ),
    iaa.BlendAlpha(
        factor=(0.2, 0.8),
        foreground=iaa.Sharpen(1.0, lightness=2),
        background=iaa.CoarseDropout(p=0.1, size_px=8)
    )
]

# Directories
input_dir = 'dataset3/myData/primer/tunnel'
output_dir = 'dataset3/myData/primer/train/tunnel_aug'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

processed_images = []

# Process each image
for filename in os.listdir(input_dir):
    if (filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".png") or filename.endswith(".PNG")
            or filename.endswith(".JPEG") or filename.endswith(".jpeg")):
        # Leer la imagen
        image_path = os.path.join(input_dir, filename)
        image = cv2.imread(image_path)

        # save OG image
        original_output_path = os.path.join(output_dir, f"original_{filename}")
        cv2.imwrite(original_output_path, image)
        processed_images.append(original_output_path)

        for augmenter in augmenters:
            augmented_image = augmenter(image=image)
            augmented_output_path = os.path.join(output_dir, f"augmented_{filename.split('.')[0]}_{augmenters.index(augmenter)}.jpg")
            cv2.imwrite(augmented_output_path, augmented_image)
            processed_images.append(augmented_output_path)

# If 1000 is not achieved continue randomly
num_images_needed = 1000 - len(processed_images)
while len(processed_images) < 1000:
    for filename in os.listdir(input_dir):
        if (filename.endswith(".jpg") or filename.endswith(".JPG") or filename.endswith(".png")
                or filename.endswith(".PNG") or filename.endswith(".JPEG") or filename.endswith(".jpeg")):
            # Leer la imagen
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Random augmentations
            augmenter = iaa.Sequential(augmenters, random_order=True)
            augmented_image = augmenter(image=image)
            random_augmented_output_path = os.path.join(output_dir, f"random_augmented_{len(processed_images)}.jpg")
            cv2.imwrite(random_augmented_output_path, augmented_image)
            processed_images.append(random_augmented_output_path)

            if len(processed_images) >= 1000:
                break

print(f"Completed and saved en {output_dir}. Total number of images: {len(processed_images)}")