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

### Custom logic

예를 들어, LLM 모델의 overconfident 성질 때문에 우리의 label에 others (기타등등) 을 추가해야 한다고 해보자. 그러면, 지금까지 학습한 모델의 마지막 층 FC layer size를 하나 늘려야 할 수 있다. 그런데, 과연 그게 최선일까? 데이터를 추가로 수집해야 할 수도 있고, 모델을 처음부터 다시 학습시켜야 할 수 있다.

이런 경우, 우리의 Service 영역에서 모델의 softmax class의 값을 확인해 특정 threshold(예를 들어, 0.9)를 넘지 못하면 (즉, 그렇게 overconfident한 LLM이 그다지 confident하지 못했다면.) others로 label을 바꿔 줄 수 있다.

> 최대한 모델과 비즈니스 로직(Product)을 분리하자.
