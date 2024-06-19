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

```python
from torchvision import tv_tensors

boxes = tv_tensors.BoundingBoxes(
    [
        [15, 10, 370, 510],
        [275, 340, 510, 510],
        [130, 345, 210, 425]
    ],
    # x좌표, y 좌표, x 좌표, y 좌표
    # 다른 선택지로 x 좌표, y 좌표, 가로 길이, 세로 길이 등이 있음
    format="XYXY", 
    canvas_size=img.shape[-2:])

transforms = v2.Compose([
   # 랜덤 자르기
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    # 밝기, 색조, 대비, 채도 및 노이즈를 조정
    v2.RandomPhotometricDistort(p=1),
    # 좌우 대칭
    v2.RandomHorizontalFlip(p=1),
])

# img 와 box 를 동시에 transform
out_img, out_boxes = transforms(img, boxes)
print(type(boxes), type(out_boxes))

plot([(img, boxes), (out_img, out_boxes)])
```

- bounding box 는 object detection을 위한 것이지만, object / semantic segmentation을 위한, mask(`torchvision.tv_tensors.Mask`), 혹은 (`torchvision.tv_tensors.Video`) 도 똑같은 방법으로 넘겨주면 된다.

## TVTensors 가 뭐죠??

- `torch.Tensor`의 subclass로, `Image`, `BoundingBoxes`, `Mask`,`Video`가 가능하다.

- 여전히 Tensor에 속하기 때문에, 평범한 Tensor 처럼 쓰면 된다.
- `sum()`, `torch.*` 등의 연산이 그대로 사용 가능하다.
- TVTensor class 들이 transforms 의 핵심이기 때문에 input을 변형하기 위해 **클래스**를 먼저 확인하고 그에 맞는 적절한 구현사항을 적용하는 것이 중요하다.

## input으로는 뭘 넣어줘야 하죠?
- 위의 예시에서는 단일 image 와, image, boxes를 동시에 넣는 것을 보았지만, 사실 임의의 구조가 다 가능하다.
- 단일 이미지, tupe, 중첩 dictionary 뭘 넣든 간에 작동하며, 넣은 input의 형태와 같은 output을 뱉어낸다.

```python
target = {
    "boxes": boxes,
    "labels": torch.arange(boxes.shape[0]),
    "this_is_ignored": ("arbitrary", {"structure": "!"})
}

# Re-using the transforms and definitions from above.
out_img, out_target = transforms(img, target)

plot([(img, target["boxes"]), (out_img, out_target["boxes"])])
print(f"{out_target['this_is_ignored']}")
```
```python
('arbitrary', {'structure': '!'})
```
- 이전의 예시에서, tuple을 넘겨주어서 결과도 tuple 이 나왔다.
- 이번에는 dictionary를 넘겨주었으니 dictionary 가 나온 것을 볼 수 있다.
- 즉, object의 **type**을 보고 그에 따른 조작을 가한다.

> 순수한 `torch.Tensor`는 대체로 `Image`로서 다뤄진다. 실제로 위의 예시 코드들에서 한번도 `torchvison.tv_tensors.Image` 클래스를 쓴적은 없으나, 첫번째 input이 box로 (Image) 였기 때문에 제대로 image transform이 이루어졌다고 보면 된다.

> 더 구체적으로는 다음과 같은 Rule을 따른다.
>> input 에 `Image`, `Video`, `PIL.Image.Image 인스턴스가 있다면, 모든 다른 pure tensor들은 그냥 패스한다.
>> 만약 없다면, 첫번째 input(pure `torch.Tensor`)을 `Image` 혹은 `Video`로 간주한다.

> 그래서, 두번째 input인 label 은 그냥 패스됐던 것이다.


## Transforms and Datasets intercompatibility

- **결국, output 데이터셋이 transform의 input과 일치해야 한다.**
- 이걸 어떻게 처리할 지는 built-in datasets를 쓰느냐, 직접 custom 한 datasets을 쓰느냐에 따라 다르다.

### built-in datasets 를 쓴다.
#### Image Classification
- 아무것도 하지 않아도 된다.
- 그냥 `ImageNet(..., transform=transforms)` 와 같이 인자로 넘겨주기만 하면 된다.
#### 그 외
- object detection, segmentation 에 쓰이는 datasets 의 경우 v2 이전에 나온 것들이라 TVTensors를 반환하지 않고 이는 문제가 된다.
- 이를 해결하기 위해 다음과 같이
`torchvision.datasets.wrap_dataset_for_transforms_v2()` 를 쓰자.

```python
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

dataset = CocoDetection(..., transforms=my_transforms)
dataset = wrap_dataset_for_transforms_v2(dataset)
```

### custom-datasets 를 쓴다.
custom-datasets 를 쓴다면, 적절한 TVTensor 클래스로 변환시켜주어야 하는데, 추천하는 변환의 타이밍은은 다음과 같다.

- dataset 클래스의 `__getitem__` 메서드 마지막에 리턴하기 직전 
- transform 파이프 라인의 시작단계