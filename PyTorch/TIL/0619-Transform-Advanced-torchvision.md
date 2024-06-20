# Transform-Advanced-torchvison

1. Tensor  
2. Dataset,DataLoader  
   1. `torch.utils.data`
3. Transform
   1. `torchvision.transforms.v2`
   2. **`torchvision.transforms`**
4. nn-model  
5. Automatic-Differentiation  
6. Parameter-Optimization  
7. model-save-load  
---

[Pytorch-torchvison-transforms](https://pytorch.org/vision/stable/transforms.html) 을 참고해 작성됨

## Start here
새롭게 추가된 v2 부터 시작하는 것이 좋다. (0618 TIL에 작성완료)  
더 많은 정보와 튜토리얼 들이 [example gallery](https://pytorch.org/vision/stable/auto_examples/index.html#gallery) 에 준비되어 있다.
  - [Transforms v2: End-to-end object detection/segmentation example](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_e2e.html#sphx-glr-auto-examples-transforms-plot-transforms-e2e-py)
  - [How to write your own v2 transforms](https://pytorch.org/vision/stable/auto_examples/transforms/plot_custom_transforms.html#sphx-glr-auto-examples-transforms-plot-custom-transforms-py)

## Input types 와 conventions
대부분의 transformation이 `PIL`과 `tensor`을 모두 지원한다. 또, `CPU`와 `CUDA` 역시 마찬가지로 모두 지원한다. 다만, 성능을 위해 `tensor`를 사용하길 권한다.
   - `v2.ToImage()`
   - `v2.ToPureTensor()`
   - `v2.PILToTensor()` 와 같은 `conversion transforms` 들을 사용해야 할 수 있다.

일반적인 `tensor`는 다음과 같은 차원을 가진다.

(`N`, `C`, `H`, `W`)
- `N` : batch size
- `C` : Channel size
- `H` : height
- `W` : width

## Dtype and expected value range

tensor의 데이터 타입에 따라 데이터의 범위가 암묵적으로 정해진다.

예를 들어, float 데이터 타입의 경우 [0, 1], `torch.uint8` 의 경우 [0, 255] 같이 정해진다.

`ToDtype` 을 이용해 dtype과 input의 범위를 변경하라.

## V1? V2?
V2를 사용할 것을 강력히 권한다. 더 빠르고, 앞으로의 업데이트도 모두 V2에서만 진행될 것이다.

### V2
- image뿐만 아니라, bounding boxes, masks, videos 도 변형이 가능하다.
- `CutMix`, `MixUp` transform 을 지원한다.
- 더 빠르다.
- 임의의 input structure에 적용 가능하다. (dicts, lists, tuples, ...)
- V2에서만 앞으로의 변경과 기능 추가 등이 일어날 것이다.

이미 V1을 쓰고 있었다면, 단순히 import문만 손봐도 바로 V2를 적용할 수 있다.

## Performance Considerations
1. v2 Transform을 사용하라.
2. PIL images 대신에 tensors를 사용하라.
3. 특히 resizing에서 `torch.uint8`을 dtype을 사용하라.
4. bilinear 혹은 bicublic mode로 resizing 하라.

```python
transforms = v2.Compose([

    # 2 PIL -> tensor
    v2.ToImage(),

    # 3 uint8으로 resizing
    v2.ToDtype(torch.uint8, scale=True),

    v2.RandomResizedCrop(size=(224, 224), antialias=True),

   # 정규화를 위해 다시 float으로 바꿔줌
    v2.ToDtype(torch.float32, scale=True),

    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

multi-processing DataLoader ( `num_workers > 0`)를 사용한다면, 이와 같은 코드가 가장 좋은 성능을 보일 것이라 한다.

> Transform 들은 input strides에 예민하게 반응하는데, 몇몇은 channels-first (C,H,W format)를, 몇몇은 channels-last (H,W,C) 형식을 선호한다.
>> 특히, `Resize` 변형은 channels-last input을 선호한다.

## classes, functionals, kernels

Transform 을 class 로도, functional 로도 할 수 있다.
(마치 `torch.nn` 과 `torch.nn.functional` 같이)

- `Resize` class 를 쓸 수도 있고, `resize()` (`torchvision.transforms.v2.functional`)를 쓸 수도 있는 것이다.

- `torchvision.transforms.v3.functional` 의 함수들에는 "kernels" 가 포함되어 있는데, 이건 bounding boxes나 masks 같은 것들을 torchscript 로 바꾸는데 특히 사용된다.
  - 이외 상황에서는 low-level 이기 때문에 딱히 볼 일 없다.


## torchscript support
Compose 대신에 torch.nn.Sequential을 써야 한다는 것은 이전에 말한 적이 있다.

functionals 들은 pure tensors (image로 간주되는) 에만 torchscript 가 지원된다. 그 이외에는 low-level kernels 를 사용해야 한다.

custom transforms 들에 `torch.jit.script`를 사용해 torchscript로 바꾸고자 한다면 그 custom 들은 `torch.nn.Module`에 기반해야 한다.

> torchscript를 안써봐서 딱히 와닿지는 않는 말들이다...


## V2-API category

- Geometry
- Color
- Composition (묶기)
- Miscellaneous (잡다)
- Conversion
- Auto-Augmentation
- CutMix - MixUp