# Convolution

1. Tensor  
2. Dataset,DataLoader  
   1. `torch.utils.data`
3. Transform
   1. `torchvision.transforms.v2`
      1. `v2.CutMix`
      2. `v2.MixUp`
   2. `torchvision.transforms`
4. nn-model  
   1. `torch.nn`
        1. Convolution Layers
            1. Convolution
            2. **`nn.Conv#d`**
5. Automatic-Differentiation  
6. Parameter-Optimization  
7. model-save-load  
---
[Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d)을 비롯한 Convolution Layers 에 속하는 api 문서를 참고해 작성됨

## `nn.Conv#d`

### `args`

- `in_channels` : input image의 채널 개수 (컬러면 3, 흑백이면 1)
- `out_channels` : output image의 채널 개수
- `kernel_size`
- `stride=1`
- `padding=0`
- `dilation=1`
- `groups=1`
- `bias=True`
- `padding_mode='zeros'`


## `nn.ConvTranspose#d`

## `nn.LazyConv#d`

## `nn.Unfold`

## `nn.Fold`
