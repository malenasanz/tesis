import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


class MNISTBinary(Dataset):
    def __init__(self, root, train=True, transform=None, digit_a=0, digit_b=1,
                 patch_h=4, patch_w=4, noise_std=0.0, max_rotation=0, seed=0,
                 patch_row=0, patch_col=0, patch_color=(255, 255, 255),
                 patch2_row=None, patch2_col=None, patch2_color=None,
                 bg_color=None, bg_threshold=30):
        base = MNIST(root=root, train=train, download=True)
        mask = (base.targets == digit_a) | (base.targets == digit_b)
        raw_labels     = base.targets[mask].numpy()
        self.images    = base.data[mask].numpy()           # (N, 28, 28) uint8
        self.labels    = (raw_labels == digit_b).astype(int)  # digit_a→0, digit_b→1
        self.transform = transform
        self.has_patch = np.zeros(len(self.labels), dtype=bool)
        self.patch_h     = patch_h
        self.patch_w     = patch_w
        self.noise_std   = noise_std
        self.patch_row   = patch_row
        self.patch_col   = patch_col
        self.patch_color  = patch_color
        self.patch2_row   = patch2_row
        self.patch2_col   = patch2_col
        self.patch2_color = patch2_color if patch2_color is not None else patch_color
        self.bg_color     = bg_color      # si está seteado, colorea el fondo en vez del patch
        self.bg_threshold = bg_threshold  # píxeles por debajo de este valor son "fondo"
        rng = np.random.default_rng(seed)
        self.angles = rng.uniform(-max_rotation, max_rotation, size=len(self.labels)) if max_rotation > 0 else None

    def set_patches(self, indices):
        self.has_patch[:] = False
        self.has_patch[indices] = True

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx].copy().astype(np.float32)

        # 1. rotar
        if self.angles is not None:
            pil = Image.fromarray(img.astype(np.uint8), mode='L')
            pil = pil.rotate(self.angles[idx], fillcolor=0)
            img = np.array(pil).astype(np.float32)

        # 2. ruido
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, img.shape)
            img   = np.clip(img + noise, 0, 255)

        # 3. convertir a RGB
        img_rgb = np.stack([img, img, img], axis=-1).astype(np.uint8)  # (28, 28, 3)

        # 4. señal espuria: fondo coloreado o patch rectangular
        if self.has_patch[idx]:
            if self.bg_color is not None:
                background = img < self.bg_threshold
                img_rgb[background] = self.bg_color
            else:
                img_rgb[self.patch_row:self.patch_row + self.patch_h,
                        self.patch_col:self.patch_col + self.patch_w] = self.patch_color
                if self.patch2_row is not None:
                    img_rgb[self.patch2_row:self.patch2_row + self.patch_h,
                            self.patch2_col:self.patch2_col + self.patch_w] = self.patch2_color

        img_pil = Image.fromarray(img_rgb, mode='RGB')
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, int(self.labels[idx])
