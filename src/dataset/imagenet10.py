import os
import numpy as np
from typing import Dict, Tuple
from PIL import Image
from continuum.datasets.base import _ContinuumDataset
from continuum.scenarios import ClassIncremental


class ImageNetSubset10(_ContinuumDataset):  
    """
    Lightweight loader for a user-defined 10-class ImageNet subset, compatible with Continuum.

    This class loads selected ImageNet classes (specified via class_name → WNID mapping)
    and stores all images in memory as resized NumPy arrays. It is primarily used to build
    class-incremental continual learning scenarios.

    Attributes:
        data_path (str): Path to the ImageNet split ('train' or 'test').
        class_names_to_wnid (dict): Mapping from human-readable class names to ImageNet WNIDs.
        class_name_to_label (dict): Mapping from class names to 0-based integer labels.
        image_size (int): Target size (H = W) to resize all images to.
    """
    def __init__(self, data_path: str, class_names_to_wnid: dict, train: bool = True, image_size: int = 224):
        self.data_path = os.path.join(data_path, "train" if train else "test")
        self.class_names_to_wnid = class_names_to_wnid
        self.class_name_to_label = {name: i for i, name in enumerate(class_names_to_wnid.keys())}
        self.image_size = image_size
        self._x, self._y, self._t = None, None, None
        self._bboxes = None

    def _setup(self) -> None:
        """
        Load the dataset into memory.

        This method iterates through all WNID folders, loads and resizes images,
        converts them to NumPy arrays, and stores them along with their labels.

        Raises:
            Exception: If an image file cannot be opened or processed.
        """
        all_x, all_y = [], []
        print(f"Loading subset from: {self.data_path}")

        for class_name, wnid in self.class_names_to_wnid.items():
            label = self.class_name_to_label[class_name]
            folder = os.path.join(self.data_path, wnid)

            if not os.path.isdir(folder):
                print(f"Missing folder for '{class_name}' ({wnid})")
                continue

            img_files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]

            for path in img_files:
                try:
                    img = Image.open(path).convert("RGB")
                    img = img.resize((self.image_size, self.image_size))
                    all_x.append(np.array(img, dtype=np.uint8))
                    all_y.append(label)
                except Exception as e:
                    print(f"Could not load {path}: {e}")

        self._x, self._y, self._t = np.array(all_x), np.array(all_y), np.zeros(len(all_x))
        print(f"Loaded {len(all_x)} images across {len(set(all_y))} classes.")


    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retrieve the full dataset (images, labels, tasks).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - x: Image data of shape (N, H, W, C).
                - y: Integer class labels.
                - t: Integer task labels (all zeros).
        """
        if self._x is None:
            self._setup()
        return self._x, self._y, self._t

    def get_sample(self, index: int) -> Tuple[np.ndarray, int, int]:
        """
        Retrieve a single sample by index.

        Args:
            index (int): Index of the sample.

        Returns:
            tuple:
                - img (np.ndarray): Image array of shape (H, W, C).
                - label (int): Class label.
                - task (int): Task label (always 0).
        """
        path = self._x[index]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        label = self._y[index]
        task = self._t[index]
        return img, label, task


def build_imagenet_subset10_scenario(
    data_path: str,
    class_names_to_wnid: Dict[str, str]
) -> ClassIncremental:
    """
    Build a Class-Incremental learning scenario using the ImageNetSubset10 dataset.

    Args:
        data_path (str): Path to the ImageNet root directory.
        class_names_to_wnid (Dict[str, str]): Mapping from class names to WNIDs.

    Returns:
        ClassIncremental: A scenario where classes are revealed in increments of size 2.
    """
    dataset = ImageNetSubset10(data_path=data_path, class_names_to_wnid=class_names_to_wnid, train=True)
    scenario = ClassIncremental(dataset, increment=2)
    return scenario
