# Tutorial

1. Tensor  
2. **Dataset,DataLoader**  
3. Transform  
4. nn-model  
5. Automatic-Differentiation  
6. Parameter-Optimization  
7. model-save-load  

## Datasets & DataLoaders

`Dataset`은 data와 label을 저장하고, `DataLoader`은 dataset 을 열거형으로 접근할 수 있게 만들어준다.

`torch.utils.data.Dataset` 에서 기본 제공되는 데이터들과 특정 데이터에 대해 작성된 함수들을 사용할 수 있다.

모델의 프로토타입을 만들거나, 벤치마킹 용도로 많이 써달라!

- [Image Datasets](https://pytorch.org/vision/stable/datasets.html)
- [Text Datasets](https://pytorch.org/text/stable/datasets.html)
- [Audio Datasets](https://pytorch.org/audio/stable/datasets.html)

### FashionMNIST 데이터를 예시로 들었다.

```python
class FashionMNIST(MNIST):
    """`Fashion-MNIST <https://github.com/zalandoresearch/fashion-mnist>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where ``FashionMNIST/raw/train-images-idx3-ubyte``
            and  ``FashionMNIST/raw/t10k-images-idx3-ubyte`` exist.

        train (bool, optional): If True, creates dataset from ``train-images-idx3-ubyte``,
            otherwise from ``t10k-images-idx3-ubyte``.
            
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

        transform (callable, optional): A function/transform that  takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    mirrors = ["http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"]

    resources = [
        # train data
        ("train-images-idx3-ubyte.gz", "8d4fb7e6c68d591d4c3dfef9ec88bf0d"),
        # train label
        ("train-labels-idx1-ubyte.gz", "25c81989df183df01b3e8a0aad5dffbe"),
        # test data
        ("t10k-images-idx3-ubyte.gz", "bef4ecab320f06d8554ea6380940ec79"),
        # test label
        ("t10k-labels-idx1-ubyte.gz", "bb300cfdad3c16e7a12a480ee83cd310"),
    ]

    # 친절하게 class 들도 적어주셨다.
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```
```python
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)
```

- `root`는 무조건 있어야 하는 디렉토리 경로를 의미한다. "./" 를 앞에 안붙여도 된다.
- `download`가 `True`이면, 데이터가 없으면 다운받고 있으면 다운받지 않는다.
- 데이터가 없어서 다운받는다면, 자동으로 `root` 이름의 폴더를 생성해준다.
  
- `train`은 `True`이면, 미리 지정된 train resource에서 데이터를 가져오고,
- `False`이면, 미리 지정된 t10k resource에서 데이터를 가져온다. (test dataset 이거나, 전체 dataset인 듯 하다.)

- `transform`이 뭔가 싶어서 찾아봤는데,
- 말 그대로 데이터에 변형을 주는 것을 의미한다.
- 예시로 든 `ToTensor()`는 단순히 Tensor 자료형으로 바꿔주는 것이고,
- 일반적으로 많이 쓰는 `PIL (Python Image Library)` 로도 바꿀 수 있는 것 같다.
- 단순히 이런 자료형으로의 변환부터
- 이미지를 뒤집고, 자르고, 색을 변형하고, 모양을 변형하는 등의 여러 조작들이 `torchvision.transforms`에 잘 구현되어 있다.
- (내부 문서에 작성된 RandomCrop의 경우 이미지의 특정부분을 잘라내는 것이다.)
---

## Creating Custom Dataset

커스텀 Dataset을 만드는 방법에 대해 서술한다.  
세가지 함수를 구현해야 한다.
- `__init__` : 생성, 초기화
- `__len__` : 데이터셋의 길이
- `__getitem__` : 데이터셋에서 데이터를 index에 따라 가져오기

다음은 커스텀 클래스를 만드는 예시이다.  
`torch.utils.data.Dataset`을 상속받아야 한다.

```python
import os
import pandas as pd
from torchvision.io import read_image

# Dataset 클래스를 상속한다.
class CustomImageDataset(Dataset):

    # annotations_file 은 csv 파일로, 다음과 같이 생겼다고 가정한다.
    """
    tshirt1.jpg, 0
    tshirt2.jpg, 0
    ......
    ankleboot999.jpg, 9
    """

    # img_dir 은 데이터의 위치이다.

    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # self.img_labels.iloc[idx,0] 에는 이미지 이름이 들어있고,
        # self.img_labels.iloc[idx,1] 에는 이미지 라벨이 들어있다.
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])

        # tensor로 바꿔준다.
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]

        # None 이 아니라면, 즉 transform을 지정해줬다면, 적용시켜 변형해준다.
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        # tuple 형태로 데이터와 라벨 한쌍을 반환해준다.
        return image, label
```

## DataLoader를 이용해 batch로 데이터 불러오기

```python
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
```

```python
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
```

- `next(iter(dataloader))` 형식으로 사용하면 된다.
- `squeeze()` 함수는 차원이 1인 차원을 모두 제거한다. 차원 arg를 지정해주면 해당 차원만 제거한다.
- `unsqueeze(dim)` 함수는 dim에 차원이 1인 차원을 하나 만들어준다.
- `squeeze()` 함수를 쓸 때 특히 batch 차원이 지워지지 않도록 주의해야 한다.