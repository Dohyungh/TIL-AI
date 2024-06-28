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
               1. **`nn.Conv#d`**
5. Automatic-Differentiation  
6. Parameter-Optimization  
7. model-save-load  
---
[Conv1d](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d)을 비롯한 Conv2d, Conv3d api 문서를 참고해 작성됨

## `nn.Conv#d`

### `args`

- `in_channels` : input image의 채널 개수 (컬러면 3, 흑백이면 1)
- `out_channels` : output image의 채널 개수 -> **다음 layer의 in_channels와 맞아야 함**
---
#### Kernel의 개수는 몇개일까?
[Confusion about the output channels of convolution neural network](https://stackoverflow.com/questions/66547536/confusion-about-the-output-channels-of-convolution-neural-network)

kernel의 경우 1 input channel, 1 output channel 당 1개가 존재한다. 만약, `Conv2d(6, 16, (3,3))`을 사용하게 되면,

kernel의 경우 6 * 16 = 96개가 되고,
하나당 (3*3) 크기의 kernel을 사용하므로 총 9 * 96 = 864 개의 파라미터를 사용하게 된다.

---
- `kernel_size` : kernel의 크기, `int` or `tuple`
- `stride=1` : kernel이 진행하는 보폭, 없으면 1칸씩
- `padding=0` : padding 값을 얼마로 할 것이냐  
padding을 tuple 로 주게 되면 길이가 2일 경우 상하/좌우 , 4일 경우 상/좌/하/우 로 입력된다.  
padding을 string으로 줄 수도 있는데, 옵션은 `"same"`, `"valid"`이 있다.
  - `same`
: ouput size 와 input size 가 동일하게 padding을 자동으로 해 줌
  - `valid`
: padding을 하지 않음.

---

#### padding과 연산 참여 횟수
`padding`과 `stride`는 각 원소의 `연산 참여 횟수`에 관여한다.
사용가능한 input에 나오지 않은 `full` padding의 경우 모든 원소가 동일한 횟수로 연산에 참여하도록 해준다. 결과적으로 (`filter_size - 1`)의 padding을 상,하,좌,우로 하게 된다.
`same` padding의 경우 이 `full` padding의 절반의 패딩을 적용하게 되기 때문에 (`filter_size-1)/2`) `half` padding이라고도 부른다.

가장 많이 사용하는 것이 `same` padding이라고 하는데, 이미지 데이터의 중요 데이터가 가운데에 몰려 있어 edge의 데이터가 연산에 덜 참여하는 것이 오히려 자연스럽기도 하고,
크기를 유지한다는 특성이 가져오는 이점도 있기 때문인 듯 하다.

---

- `padding_mode='zeros'` : padding에 뭘 채울 것이냐
   - `zeros` : 0으로 채운다.
   - `reflect` : 원본 데이터를 반전한다. 당연히 padding에 제한이 생긴다.
   - `replicate` : 맨 끝 데이터를 반복한다.
   - `circular` : 주어진 데이터의 순환형태가 되도록 padding 해준다.
- `dilation=1` : kernel 원소간의 간격
- `groups=1` : input과 output 간의 연결을 조절한다. `in_channels`와 `out_channels` 모두 `groups`로 나누어져야 한다.
  1. 원 데이터를 group으로 쪼개서
  2. kernel에 forwarding 한 후
  3. 결과를 concat
  4. input이 결국, 서로 완벽히 분리되어
  5. 완벽히 분리된 (concat만 됐을 뿐인) output을 내놓게 된다.
   - `groups` == `in_channels` && `out_channels` ==`양의 정수 * in_channels` 일 때를 "`depthwise convolution`" 이라고 한다.
