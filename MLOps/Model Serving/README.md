# Model Serving

[MLOps: Overview](<../../Paper/240723-Machine-Learning-Operations(MLOps)-Overview-Definition-And-Architecture/README.md>) 에서 model serving은 REST API의 형식으로 이루어지며, Normal Latency(보통 응답시간) / Batch / Offline store 혹은 Low Latency(빠른 응답시간) / Real-time / Online store 를 사용한다는 언급이 나온 적이 있었다.

이를 더 자세하게 파고들어 ML 모델을 어떻게 어플리케이션에서 사용하는지를 공부해본다. 특히, 제공하고자 하는 서비스가 위의 Low / Normal Latency 두 가지 경우에 대해서 어떤 방식에 더 적합한지를 구분할 수 있도록 하자.

딥러닝 모델의 경우 그 모델의 크기 자체가 Challenge가 된다. 분산 GPU 방식, 클라우드 서비스 등 이름만으로 생소한 것들에 대해 대략적인 감을 잡고, ChatGPT와 같은 거대 언어 모델의 Serving은 어떻게 이루어 지고 있는지 이해하는 것을 목표로 하자.

> 순서대로 다음의 세 자료를 읽고 느낀점을 정리하였습니다.

- [Model-Serving : MadewithML](https://madewithml.com/courses/mlops/serving/)

- [새로운 루다를 지탱하는 모델 서빙 아키텍처 — 3편: 안정적인 LLM 서비스를 위한 서빙 최적화 기법](https://tech.scatterlab.co.kr/serving-architecture-3/)

- [Full-stack-deeplearning : Lecture 5 - Deployment](https://fullstackdeeplearning.com/course/2022/lecture-5-deployment/)

---

## 모델의 요구사항을 이해하자

모델의 성능이 마음에 들고, 이제 이 모델로 어플리케이션을 만들기로 마음 먹었다면, 이제 모델을 어떻게 serving 할 지 결정해야 한다. 우리의 서비스가 방금 들어온 클라이언트의 요청에 대해 실시간으로 반응해 결과를 내놓아야 한다면, 우리의 모델은 **High throughput, Low Latency, Online inference** 를 강요받는다. 반면에, 하루의 끝에 한 번 결과를 내놓아야 한다면, (예를 들어, 하루의 주식 시장을 정리해 다음 날의 동향을 예측한다거나) **Low throughput, Normal Latency, Offline inference** 를 만족하면 된다.

사실, 모델의 Scalability (확장 가능성)과, Flexibility (유연성)는 높을 수록 좋은 것이 당연하다. 하지만, 그 모든 상황에 적응할 수 있는 **모델**을 만드는 것보다는, 우리의 서비스가 제공하는 **AI 기능을 앞에서 언급한 Latency에 따라 구분해 서로 다른 파이프라인을 거치게 하는 것**이 더 쉽고, 현실적인 방법일 수 있다.

이에 따라 각각의 AI 기능을 REST API의 각 자원에 매핑시켜 각각의 요청에 대해 두 가지 파이프라인 중 하나에 들어가도록 설계하는 것이 구체적인 방안이 되겠다.

## Frameworks

어떤 프레임워크를 선택해야 할지 고민할 때 도움이 될 수 있는 몇가지 고려사항을 적었다.

- **Learning Curve** : 모델 서빙을 위해서 새로운 프레임워크를 배워야 하는가? 혹은 배워야 할 정도로 해당 프레임워크가 가치있는가?
- **Framwork agnostic** : Pytorch 든, TensorFLow 든 상관없는 서비스를 만들자.
- **Scale** : 스케일링이 그냥 config 파일을 수정하는 것만큼이나 쉬워야 한다.
- **Composition** : 여러 모델과 비즈니스 로직을 서비스에 적용 가능하다.
- **Integrations** : 인기있는 API 프레임워크와 통합 가능하다. (예시로 FastAPI, LangChain, Kubernetes를 들었다.)

위 사항을 모두 고려해보았을 때 선택할 수 있는 프레임워크는 다음과 같다.

- **Ray Serve** : [madewithml](https://madewithml.com/courses/mlops/serving/)의 선택
- Nvidia Triton
- HuggingFace
- Bento ML

모델을 불러올 때에는 다음과 같은 사항을 고려해야 한다.

- Batch or Real-time
- Batch Size
- Version of Model (run으로 구분했다면 run-id, checkpoint 라고도 함)
- **Custom logic**

## Batching 전략

Batching 자체만 하더라도 많은 최적화 serving 전략들이 있다. 가장 간단하게는 **batch size**(얼만큼 input이 쌓였을 때 처리할지)와 **max_latency**(얼만큼만 기다릴지)를 input의 빈도에 맞추어 설정해주어야 한다. 또한 내부적으로는 배치를 실행할 때, 배치 차원을 제외한 나머지 shape을 맞추어야 하기 때문에(정확히는 seq_len) **버킷팅 전략** (길이가 비슷한 seq 끼리 묶는다.)을 사용할 수도 있고, padding을 통해 seq_len 을 맞추고, 디코더의 마스킹을 통해 이를 공백으로 처리할 수 있다.

### Iteration Batching

나의 짧은 문장이 엄청나게 긴 문장과 묶이는 바람에 한참을 기다린다면 아주 화가 날 것이다.
이를 해소하기 위해 Iteration Batching은 사용자에게 새롭게 수신한 Input이 있다면 Batch에 즉각적으로 포함시켜서 (~~쉽게 말해 막차 태운다~~), 그리고 생성이 끝난 요청이 있을 경우 응답을 즉시 반환함으로서 클라이언트의 대기시간을 최소로 한다. (~~이미 특허등록 된 기술이라고 한다~~)

### 멀티 턴

[HyperClova](https://engineering.clova.ai/posts/2022/03/hyperclova-part-3)가 발견했지만 직접 사용하지는 않은 것으로, 모델이 가용한 범위 내에서 이전에 등장한 토큰들을 모두 참조해 다음 토큰을 계산하게 되는 트랜스포머의 특징을 이용했다. 이전까지 등장한 토큰에 대한 계산결과를 캐싱해놓는데, 이때 조막만한 GPU의 메모리에 저장하는건 불가능하고, DRAM 에 데이터를 저장하는 과정에서 오히려 먼 곳을 찍고 오게 되면서 Trade-off가 발생했다고 한다.

## FasterTransformer

[CLOVA Engineering BLOG](https://engineering.clova.ai/posts/2022/01/hyperclova-part-1)에 따르면, 모델 학습을 위한 프레임워크와 모델 추론(inference)를 위한 프레임워크가 동일한 것에 분명 장점이 있지만, 모델의 크기가 커지면 커질수록 추론에 학습이 필요없다는 점을 최대한 활용해야 할 수도 있어 보인다. 즉, 추론만을 위해 이전과는 다른 프레임워크를 선택하는 것도 고려해야 한다. LLM 모델을 위한 FasterTransformer가 바로 그것이다. FasterTransformer의 가속 원리는 모델을 병렬화 (텐서와 파이프라인에서)하는 것이다. 역시나 python이 아닌 C++/CUDA로 작성됐으며, TensorFlow, Pytorch, Triton Backend API를 제공한다.

[NVIDIA Technical BLOG](https://developer.nvidia.com/ko-kr/blog/increasing-inference-acceleration-of-kogpt-with-fastertransformer/)에서 그 구체적인 가속 방법을 소개한다.

- Layer Fusion : 여러 레이어를 단일 레이어로 결합한다.
- Multi-head attention Acceleration : 시퀀스에서 토큰 간의 관계를 연산하는데에 메모리 복사와 연산이 많이 필요하다. 이를 캐싱과 퓨즈커널로 최소화한다.

  [ScatterLab BLOG](https://tech.scatterlab.co.kr/serving-architecture-3) 에서 Key/Value Caching 에 대한 언급이 나오는데, `.generate(..., use_cache=True)`의 형식으로 간단히 수행하는 것을 보아 목표하고자 하는 바는 비슷했던 것 같다.

- GEMM Kernel autotuning : matmul 연산의 매개변수(Attachment 레이어, 크기 등)을 하드웨어 수준에서 실시간 벤치마크를 통해 최적화한다.

- Lower precision: FP16, BF16, INT8 과 같은 낮은 정밀도의 데이터 타입으로 더 빠른 연산을 할 수 있다.

  그러나 HyperClova 팀은 이것 때문에 오히려 오버플로우로 인한 버그를 겪어서 다시 FP32 자료형으로 돌아갔다고 한다.

## Custom logic

예를 들어, LLM 모델의 overconfident 성질 때문에 우리의 label에 others (기타등등) 을 추가해야 한다고 해보자. 그러면, 지금까지 학습한 모델의 마지막 층 FC layer size를 하나 늘려야 할 수 있다. 그런데, 과연 그게 최선일까? 데이터를 추가로 수집해야 할 수도 있고, 모델을 처음부터 다시 학습시켜야 할 수 있다.

이런 경우, 우리의 Service 영역에서 모델의 softmax class의 값을 확인해 특정 threshold(예를 들어, 0.9)를 넘지 못하면 (즉, 그렇게 overconfident한 LLM이 그다지 confident하지 못했다면.) others로 label을 바꿔 줄 수 있다.

> 최대한 모델과 비즈니스 로직(Product)을 분리하자.
