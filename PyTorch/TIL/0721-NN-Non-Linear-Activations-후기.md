# Non-linear Activations (weighted sum, nonlinearity)

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
            1. `nn.Conv#d`
            2. `nn.ConvTranspose#d`
      2. Pooling Layers
      3. **Non-Linear Activations**
         1. **후기**
5. Automatic-Differentiation
6. Parameter-Optimization
7. model-save-load

---

# 후기

지난했던 `torch.nn`의 활성함수 정리가 끝났다. (0701~0721)

`multi-head-attention` 은 transformer 문서에서 더 자세히 공부한 다음 적을 계획이다. (~~언제?~~)

어느 정도 활성함수의 변천사와 활성함수가 가져야할 긍정적인 특징과 문제점들, 그리고 그 문제점을 개선하기 위한 시도들에 대해서 이해해 볼 수 있었다. 그 과정에서 개념들이 복잡하게 얽혀있기도 하고 여기저기 파편적으로 퍼져있었기도 해서 더 어려웠던 것 같다. 중간 중간 대학교에서 배웠던 것들이 나올 때마다(MLE, CrossEntropy, vanishing gradient, Convolution) 미리. 제대로. 공부해놓지 않은 자신이 한탄스럽기도 했다.

이제 순서대로라면 `normalization layer`를 작성해야 겠지만,, SSAFY 특화 프로젝트를 위해 MLOps로 살짝 눈을 돌려서 여러 미들웨어와 데이터 적재, 처리, 가공, 학습 과정을 공부해보려고 한다. LLM 을 소재로 한 프로젝트를 계획하고 있는데, 컴퓨터의 사양과 현재 나의 지식 수준으로 가능한 스케일이 어느 정도인지 가늠이 되지 않아서 걱정이다.
