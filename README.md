# Asian Subcontinent Dataset (ASCD)

The **Asian Subcontinent Dataset (ASCD)** is a multisensor dataset collected from different countries in the Asian subcontinent, specifically India, Bangladesh, and Sri Lanka. The dataset consists of remote sensing images from two primary sources: **Google Earth Pro** and **IKONOS-2 satellite images**. ASCD contains **14 land cover classes** that represent various geographical features and man-made structures. The dataset includes **1674 images** with a size of **227 × 227 × 3** (width × height × RGB channels).

This dataset is ideal for training and evaluating machine learning models in areas like land cover classification, environmental monitoring, and remote sensing.

## Dataset Overview

The dataset is divided into **14 land cover classes**, and each class contains a collection of images.

### Dataset Structure:
/ASCD/
  ├── /Agricultural/
  ├── /Desert/
  ├── /Port/
  ├── /Airport/
  ├── /Forest/
  ├── /River/
  ├── /Beach/
  ├── /Freshwater/
  ├── /Runway/
  ├── /Bridge/
  ├── /Mediumresidential/
  ├── /Seawater/
  ├── /Denseresidential/
  └── /Playground/
  


### Description of Each Land Cover Class:

1. **Agricultural**: Images representing agricultural fields, crops, and farmlands.
2. **Desert**: Images of dry, arid regions with sparse vegetation.
3. **Port**: Images of seaports or harbor areas.
4. **Airport**: Images of airports, runways, and airstrip areas.
5. **Forest**: Forested areas with a significant presence of trees and vegetation.
6. **River**: Images of rivers, streams, and freshwater bodies.
7. **Beach**: Coastal areas with sandy beaches.
8. **Freshwater**: Ponds, lakes, and other freshwater bodies.
9. **Runway**: Images of runways used for aviation.
10. **Bridge**: Images of bridges or overpasses.
11. **Medium Residential**: Residential areas with a medium population density.
12. **Seawater**: Coastal water bodies such as oceans and seas.
13. **Denser Residential**: Densely populated urban residential areas.
14. **Playground**: Parks, open spaces, and recreational areas.

Each class folder contains a number of images ranging from 100 to 204 images, providing a total of **1674 images**.

### Image Format:

- Each image is of size **227 × 227 × 3** (width × height × RGB channels).
- Images are stored in common formats like **JPEG**.

This structure allows easy access to images based on their land cover class and is suitable for training machine learning models for classification tasks.

## Dataset Usage

### **Loading the Dataset**

To use the dataset, you can load it using **PyTorch** or **TensorFlow**. Below is an example using **PyTorch**:

```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ASCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(os.path.join(root_dir, "images"))
        self.image_paths = []
        for class_name in self.classes:
            class_folder = os.path.join(root_dir, "images", class_name)
            self.image_paths += [(os.path.join(class_folder, f), class_name) for f in os.listdir(class_folder)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        return img, label
