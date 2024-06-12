# PyTorch 2.0


- 1.13 버전에서 2.0 으로 버전 업
- 이전 버전 코드들 그대로 사용 가능
- 추가된 주요 기능으로 다음의 4가지 기능으로 뒷받침되는 `torch.compile` 을 강조
  - TorchDynamo(프로그램 안정성), AOTAutograd(autograd 엔진에 기능 추가), PrimTorch(2000개 연산자 -> 250개로 줄임), TorchInductor(딥러닝 컴파일러)
  - TIMM, TorchBench, Huggingface 에 있는 모델들 가지고 테스트 한 결과 속도 향상됨
> **Accelerator (가속기)**  
> 
> Accelerator에 대한 언급이 많이 나와 조사해보니, 모델 학습을 더 효율적으로 할 수 있도록 해주는 하드웨어나 시스템 클래스를 의미한다.
> GPU가 DL에 효율적이라는 말이랑 일맥상통하는 듯하다.
> 최근에는 ASIC (Application-Specific Integrated Circuit)이라는 새로운 가속기가 각광이다.
- 어쩔 수 없이 가속기의 등장과 성능에 대한 욕심 때문에 Python으로 개발하고 싶은데 C++ 로 작업해야 하는 일이 생겼나 보다.
- 이런게 접근성을 떨어뜨리는게 상당히 마음에 안드시는 듯.
- Graph Acquisition
- Graph Lowering
- Graph Compilation
- Graph란??


참고 링크:   
[PyTorch](https://pytorch.org/get-started/pytorch-2.0/)
[JunYoung's blog](https://junia3.github.io/blog/pytorch2)