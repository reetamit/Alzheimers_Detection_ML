import os
import pandas as pd
from PIL import Image
import io
import base64

# 🔧 CONFIGURATION
image_folder = "./data/images"  # Folder containing PNG images
output_parquet = "./data/MRItrain.parquet"

# Optional: define label mapping based on filename or folder structure
def infer_label(filename):
    # Example: if filenames are like 'cat_001.png', 'dog_002.png'
    return filename.split('_')[0]

# 📦 Collect image data and labels
data = []
for fname in os.listdir(image_folder):
    if fname.lower().endswith('.png'):
        fpath = os.path.join(image_folder, fname)
        with Image.open(fpath) as img:
            # Convert image to bytes
            with io.BytesIO() as output:
                img.save(output, format='PNG')
                img_bytes = output.getvalue()
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        label = infer_label(fname)
        data.append({
            "filename": fname,
            "image_base64": img_base64,
            "label": label
        })

# 📝 Save to Parquet
df = pd.DataFrame(data)
df.to_parquet(output_parquet, engine='pyarrow', index=False)
print(f"Saved {len(df)} images to {output_parquet}")
