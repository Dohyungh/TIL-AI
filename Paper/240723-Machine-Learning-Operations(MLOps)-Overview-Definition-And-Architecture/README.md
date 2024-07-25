# Machine Learning Operations(MLOps) : Overview, Definition, and Architecture

- Dominik Kreuzberger
- Niklas Kuhl
- Sebastian Hirschl

- 카를스루에 공과대학교 (에드워드 텔러-수소폭탄, 카를 벤츠-자동차 벤츠) 에서 arxiv에 2022년에 게재한 논문

## abstact

> 초록은 직역

모든 산업계 ML 프로젝트의 목표는 ML 제품을 만들어서, 빠르게 런칭하는 것이다. 그러나 ML 제품을 자동화하고, Operationalize 하는 것은 대단히 어려운 일이기에, 많은 노력과 노고가 기대에 못 미치고 끝나는 경우가 많다. Maching Learning Operations (이하 MLOps)의 패러다임은 이 문제를 다룬다.

MLOps에는 여러 측면이 있는데, 최선의 실무(실천)들, 개념 집합들, 개발 문화들이 바로 그들이다.
그러나, 여전히 모호한 용어가 많고, 연구자들과 전문가들의 용어에 대한 결론(Consequences) 마저 다르다. 이 gap을 줄이기 위해 우리는 mixed-method 리서치를 수행했는데, 여기에는 문헌 리뷰, 툴(tool) 리뷰, 전문가 인터뷰가 포함된다.

이 연구의 결과로, 우리는 필수 원칙들, 구성요소와 역할들에 대해서 통합된 개요를 여러분께 제공한다. 이에 더해서 MLOps 의 정의를 다듬어서 필드에서 벌어지고 있는 무한한 도전들을 강조하려 한다.

결론적으로, 이 작업은 ML 연구자들과 ML 제품을 자동화하고 운영하고자 하는 실무자들에게 명확한 기술들이 포함된 가이드라인을 제공할 것이다.

`CI/CD`, `DevOps`, `Machine Learning`, `MLOps`, `Operations`, `Workflow Orchestration`

## 본문

### Introduction

> MLOps 란 무엇인가?

위 질문에 대한 답을 찾기 위해

1. MLOps의 중요한 원칙들을 찾고,
2. 그 핵심 구성요소들을 발라낸 다음,
3. 필요한 역할들을 강조하며,
4. 보편적인 MLOps의 시스템 디자인을 얻어낸다.

이후의 내용들은 다음과 같이 구성된다.

1. 필요한 기초 요소들, 관련된 연구들에 대해 상세히 살펴본다.
2. 이용가능한 방법론에 대한 개요를 제공한다. (문헌, tool, 인터뷰)
3. 이 방법론의 적용으로 얻을 수 있는 통찰에 대해 제시하고, 정의한다.
4. 요약, 제약사항, 제언하고 마친다.

### Foundations of DevOps

전통적인 waterfall, agile 방법론들은 결국 비슷한 목적을 가지고 있는데, **즉각 제공가능한 소프트웨어 제품을 만들자** 는 것이다. 2008~2009 년에는 "**DevOps**" 개념이 생겨났고, 소프트웨어 개발의 문제들을 경험적으로 많이 해결해줬다.

DevOps는 단순한 방법론을 넘어 사회 / 기술적 문제들을 다루는 패러다임을 의미한다.

개발과 운영사이의 gap을 없애고 협업, 소통, 지식의 공유를 강조한다. 자동화와 지속통합/배포로 빠르고, 잦지만 신뢰도 높은 배포를 보장하는데 더해 지속 테스팅, 품질 관리, 지속적 모니터링, 로깅(로그를 남기는 것), 피드백 루프이 이루어진다.

DevOps Tool 들은 크게 다음의 6종류로 구분된다.

1. 협업과 지식 공유

   - (Slack, Trello, GitLab wiki)

2. 코드 관리

   - (GitHub, GitLab)

3. 빌드

   - (Maven)

4. CI

   - (Jenkins, GitLab CI)

5. 배포 자동화

   - (Docker, Kubernetes)

6. 모니터링, 로깅
   - (Prometheus, Logstash)

이런 DevOps 방식이 많은 성공을 거뒀고, 이제 ML 에도 적용하려 한다.

### 방법론

<p align="center">
<img src="./assets/OverviewOfTheMethodology.png" style="width: 40%"/>
</p>

학술계의 인사이트, 실무진의 전문성을 모두 놓치지 않기 위해 문헌, Tool, Interview의 3가지 방법론을 모두 사용해 MLOps의 개념을 정립하고, 이후 결과 파트에서 그 내용에 대해 설명한다.

#### Literature Review

- Webster and Watson 의 논문 리뷰 방법론을 참고해 수행됨.
- Barbara Kitchenham의 [Systematic literature reviews in software engineering – A systematic literature review](https://www.sciencedirect.com/science/article/abs/pii/S0950584908001390?via%3Dihub) 도 참고함.

몇번의 검색 끝에 다음과 같이 검색어를 정의했다.

- DevOps
- CICD
- Continuous Integration
- Continuous Delivery
- Continuous Deployment
- Machine Learning
- MLOps
- CD4ML

구글 스칼라를 비롯한 여러 사이트들을 검색했는데, 사실 DevOps를 ML에 적용하는 것은 학술계에는 아직 생소한 일이다. 그래서 본 연구가 진행될 때에는 리뷰된 연구들이 몇개 되지 않았다.

그래서 non-peer-reviewed 연구들까지 모두 검색했고, 1864개의 논문을 얻었다. 특정 기준을 갖고 27개의 논문을 추려낸 결과, 그들은 모두 peer-reviewed 논문들이었다.

#### Tool Review

27개의 논문과 8개의 인터뷰를 거치고 난 후, 다양한 Open Source tools, 프레임워크, 상업적 ML 클라우드 서비스들을 리뷰했다. 어떤 기술적 구성요소들이 있는지 이해하기 위한 작업이었다.

리뷰한 tool들은 다음과 같다.

<p align="center">
<img src="./assets/ToolReview1.png" style="width: 40%"/>
<img src="./assets/ToolReview2.png" style="width: 40%"/>
</p>

#### Interview Study

- Myers and Newman 방식을 참고함 [The qualitative interview in IS research: Examining the craft](https://www.sciencedirect.com/science/article/abs/pii/S1471772706000352?via%3Dihub)

- 몇명을 인터뷰할 것인지도 결정해야 해서 이론적인 표본 접근방식을 적용했다. [The discovery of grounded theory: strategies for qualitative research]

LinkedIn을 통해서 여러 다른 기업, 배경, 성별의 MLOps 전문가들 8명을 찾아 스크립트를 작성한 후 인터뷰를 진행했다.
이때 더 이상 새로운 범주나 개념이 등장하지 않을 때까지 진행되었다.

<p align="center">
<img src="./assets/IntervieweeList.png" style="width: 40%"/>
</p>

> 인터뷰이들 목록인데, 궁금하다..!

## 결과

위의 방법론들을 사용해 중요 원리, 구성요소로의 객체화, 필수 역할등의 결과를 얻었고, MLOps의 개념과 정의를 이끌어냈다.

### Principles

보편적인 사실이나 가치, 가이드를 제공한다는 의미의 priniciple은 MLOps의 "최선의 방법"과 밀접하게 연관되어 있다. 9개의 원칙(Principle)을 얻을 수 있었다.

<p align="center">
<img src="./assets/Principles.png" style="width: 60%"/>
</p>

#### `CI/CD automation` - P1

- 지속적 통합
- 지속적 전달 (실제 product에 반영되기 위해 버튼 하나만 누르면 되는 순간까지)
- 지속적 배포 (실제 product에 자동으로 반영)

빌드, 테스트, delivery, 배포를 수행한다. 개발자에게 특정 단계의 진행이 성공인지 실패인지를 빠르게 알려줘서 전반적인 생산성이 개선된다.

#### `Workflow Orchestration` - P2

유향 비순환 그래프 (DAGs)로 ML workflow 파이프라인을 조직한다. 관계와 의존성을 고려해 일의 **순서**를 정한다.

#### `Reproducibility` - P3

ML을 실행해서 정확히 똑같은 결과를 얻는 능력을 말한다.

#### `Versioning` - P4

모델의 버전 뿐만 아니라, 데이터, 코드의 버전까지 단순히 재생산(Reproducibility)을 보장하는 것을 넘어 흐름을 짚을 수 있게 한다.(traceabiltiy)

#### `Collaboration` - P5

데이터, 모델, 코드에 대해 협업이 가능하게 한다. 기술적인 면만 말하는 것이 아니다. 협업과 소통을 통해 서로 다른 일을 하는 사람들 간의 거리를 좁혀 준다.

#### `Continuous ML training & evaluation` - P6

지속적 ML 학습이란, 주기적으로 새로운 feature 데이터에 대해 재학습시키는 것을 말한다. 이것은 `Monitoring` 컴포넌트, 피드백 루프, 자동 ML workflow 파이프라인의 지원들이 있어야 한다.  
이때 모델의 바뀐 성능을 평가하기 위해 항상 evaluation이 포함된다.

#### `ML metadata tracking/logging` - P7

metadata는 각각의 ML workflow에 대해 트랙킹되고, 로깅된다. 매 학습이 이루어질 때마다 어떤 코드, 어떤 데이터, 어떤 결과, 어떤 파라미터를 썼는지 완벽히 기록되어야 한다.

#### `Continuous monitoring` - P8

주기적으로 데이터, 모델, 코드를 평가하는 것을 의미한다. 잠재적인 에러와 변경사항들이 있는지 관찰한다.

#### `Feedback Loops` - P9

여러 개의 피드백 루프가 필요하다. 예를 들어, 모델의 실험에서 이전의 feature 엔지니어링 단계로의 루프나 `Monitoring` component에서 재학습을 위해 스케줄러로의 루프가 있겠다.

### Technical Components

Principle을 알아낸 후, 정확한 component들과 구현에 대해 설명하겠다. 각각의 component들에 대해 필수적인 **기능**을 열거한다. 괄호안에는 각각의 기능 component가 어느 Principle component를 구현할 수 있는지를 적었다.

#### `CI/CD Component` - C1 [P1, P6, P9]

- Jenkins
- GitHub actions

#### `Source Code Repository` - C2 [P4, P5]

코드 저장과 버전관리 가능

- Bitbucket
- GitLab
- GitHub
- Gitea

#### `Workflow Orchestration Component` - C3 [P2, P3, P6]

- Apache Airflow
- Kubeflow Pipelines
- Luigi
- AWS SageMaker Pipelines
- Azure Pipelines

#### `Feature Store System` - C4 [P3, P4]

자주 쓰는 것들을 모아 놓을 수 있는 중앙 저장소이다.
