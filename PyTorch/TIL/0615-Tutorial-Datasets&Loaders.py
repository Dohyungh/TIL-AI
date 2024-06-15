import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


training_data = datasets.FashionMNIST(
    # 데이터가 있는 위치
    # 없으면, .py 파일이 있는 위치에서 "data" 폴더를 생성해 다운받는다.
    root="./data",
    train=True,
    download=False,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=False,
    transform=ToTensor()
)

# print(training_data)

# dict 형태로 지정
# 이거 사실 FashionMNIST의 classes에 배열로 저장되어있어서 불러올 수 있다.
# print(datasets.FashionMNIST.classes)
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
print("test")
print(labels_map[0])

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
# 1부터 9까지
for i in range(1, cols * rows + 1):

    # 아무래도 자기들 공식 문서이다 보니 내장된 randint를 써야했나보다..
    sample_idx = torch.randint(len(training_data), size=(1,)).item()

    # index안에 데이터와 label이 붙어 있어서 이렇게 분리시킨다.
    img, label = training_data[sample_idx]
    # 3,3 짜리 격자에서 1~9까지
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()

import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Display image and label.
for i in range(0, train_dataloader.batch_size):
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[i].squeeze()
    label = train_labels[i]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")
    print(f"Label: {labels_map[label.item()]}")