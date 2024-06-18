# Transform-Advanced-v2

1. Tensor  
2. Dataset,DataLoader  
   1. `torch.utils.data`
3. Transform
   1. **`torchvision.transforms.v2`**
   2. `torchvision.transforms`
4. nn-model  
5. Automatic-Differentiation  
6. Parameter-Optimization  
7. model-save-load  
---

[Pytorch-torchvision-transforms-v2](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_getting_started.html#sphx-glr-auto-examples-transforms-plot-transforms-getting-started-py) 를 참고해 작성됨.

## Setup
```python
from pathlib import Path
import torch
import matplotlib.pyplot as plt
plt.rcParams["savefig.bbox"] = 'tight'

from torchvision.transforms import v2
from torchvision.io import read_image

torch.manual_seed(1)

# If you're trying to run that on collab, you can download the assets and the
# helpers from https://github.com/pytorch/vision/tree/main/gallery/
from helpers import plot
img = read_image(str(Path('../assets') / 'astronaut.jpg'))
print(f"{type(img) = }, {img.dtype = }, {img.shape = }")
```



## The basics
Torchvison transoforms 는 `torch.nn.Module` 과 거의 비슷하게 작동한다.
transform을 객체화 해서, input을 전달하고, transform 된 결과물을 얻는다.

```python
transform = v2.RandomCrop(size=(224, 224))
out = transform(img)


# helper라는 코랩에서 ipynb 파일로 실행해 볼 수 있는 패키지이다.
plot([img, out])
```

## normal (image classification)

```python
transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
out = transforms(img)

plot([img, out])
```

- 이러한 transformation pipeline 은 `Datasets`의 `transform` arg로 전달된다. 예) `ImageNet(...,transform=transforms)`



---
`v2.Compose(transforms: Sequence[Callalbe])`
- transform 들을 배열 (`[]`)에 담아서 전달해주면,
- 여러 transform 들로 구성된 transform을 만들어준다.
- 이는 torchscript(모델 실행의 보편화를 위한 script)를 지원하지 않는다.
  - transform을 script 로 만들고 싶으면 다음과 같은 `torch.nn.Sequential`을 사용해야 한다.
  ```python
  # 이 Compose 코드 대신에
    transforms.Compose([
      transforms.CenterCrop(10),
      transforms.PILToTensor(),
      transforms.ConvertImageDtype(torch.float),
   ])
   ```
  ```python
  # 이 Sequential 코드를 사용해야 한다.
  transforms = torch.nn.Sequential(
    transforms.CenterCrop(10),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),)
    scripted_transforms = torch.jit.script(transforms)
   ```
----

## Detection, Segmentation, Videos

- v2 에는 Image Classification 이외에도 bounding boxes, segmentation, detection 관련한 transform 기능이 추가되었다.

