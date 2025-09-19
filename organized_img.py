import os
import shutil

src_dir = r"D:\Butterflies.proj\Processed_train" 
classes = ["resize","blurred","scaled", "rotated", "normalized"] 

for cls in classes:
    os.makedirs(os.path.join(src_dir, cls), exist_ok=True)


for file in os.listdir(src_dir):
    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        for cls in classes:
            if cls.lower() in file.lower():  
                shutil.move(
                    os.path.join(src_dir, file),
                    os.path.join(src_dir, cls, file)
                )
                break
print("Finished organizing images into class folders!")