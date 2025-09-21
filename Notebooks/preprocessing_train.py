import cv2
from PIL import Image
import numpy as np
import os

def load_images_from_folder(folder_path):
    images = []
    filenames = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert("RGB")  # ضمان انها 3 قنوات
                images.append(img)
                filenames.append(filename)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return images, filenames

# --- Paths ---
train_folder = r"D:\CIFAR-10_CNN\Data\train"
output_dir = r"D:\CIFAR-10_CNN\Processed_train"
os.makedirs(output_dir, exist_ok=True)

images, filenames = load_images_from_folder(train_folder)
print("Number of images loaded:", len(images))

for i, image in enumerate(images):
    img = np.array(image)  # PIL → numpy

    print(f"Processing {filenames[i]} | shape={img.shape}")

    # Resize
    scaled_image = cv2.resize(img, None, fx=0.5, fy=0.5)
    resized_image = cv2.resize(img, (224, 224))
    cv2.imwrite(os.path.join(output_dir, f"{i+1}_scaled.jpg"), scaled_image)
    cv2.imwrite(os.path.join(output_dir, f"{i+1}_resized.jpg"), resized_image)

    # Blur
    blurred_image = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(os.path.join(output_dir, f"{i+1}_blurred.jpg"), blurred_image)

    # Rotate
    height, width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 45, 1)
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))
    cv2.imwrite(os.path.join(output_dir, f"{i+1}_rotated.jpg"), rotated_image)

    # Normalize
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(image_rgb)
    r_normalized = cv2.normalize(r.astype('float32')/255.0, None, 0, 1, cv2.NORM_MINMAX)
    g_normalized = cv2.normalize(g.astype('float32')/255.0, None, 0, 1, cv2.NORM_MINMAX)
    b_normalized = cv2.normalize(b.astype('float32')/255.0, None, 0, 1, cv2.NORM_MINMAX)
    normalized_image = cv2.merge((r_normalized, g_normalized, b_normalized))
    save_img = cv2.cvtColor((normalized_image * 255).astype('uint8'), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, f"{i+1}_normalized.jpg"), save_img)

print("Finished preprocessing all images! Check the Processed folder.")
