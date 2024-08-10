# NVIDIA Triton Inference Server

NVIDIA에서 만든 추론 전용 서버이다. 많은 기능들이 이미 내장되어 있는데, [KakaoPay Tech 블로그](https://tech.kakaopay.com/post/model-serving-framework/)에서 모델 서빙용으로 선정했다기에 직접 설치해보고, get Started! 해보았다.

[Triton-Official-Tutorial](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/getting_started/quickstart.html)에 기반해 작성되었습니다.

## WSL - Ubuntu

Ubuntu LTS가 24.xx 버전까지 나왔는데 한동안의 재설치를 거치고 결국 20.xx 버전 LTS 를 사용했다.

`그래픽 카드` - `Triton Inference Server` - `CUDA` - `CUDA Tool Kit` - `Nvidia Graphics Driver`

위에 적은 모든 친구들의 버전 호환성을 확인해서 잘 설치해야 했다. (~~사실, 아직 Pytorch 버전은 확인하지 않아서 불안하다...~~)

> 중간에 lspci | grep -i nvidia 에 아무것도 안나오는데,  
> nvidia-smi 에는 RTX 4050 그래픽카드가 버젓이 잘 나와서 뭔가 문제가 있는 줄 알았다.

> 그런데 알고 보니, 원래 lspci -k 에 3d microsoft xxx 로 나오는 것이 맞다고,,Window가 제공하는 일종의 사생활 보호 같은 것이란다.. 한참을 이것 때문에 디버깅하면서 바이오스도 들어가서 PCI 버스를 확인해보고 집에 있는 데스크탑이랑 비교해보고 했는데 허무했다.

### Model 준비

- need: 모델을 어떤 형태로 준비해야 하는지 추후에 공부가 필요하다. torch-script가 가능해야 하는 듯 하다.

본 tutorial 에서 제공하는 모델들을 사용한다. 다음과 같다.

```
2024-08-11 00:28:53 +----------------------+---------+--------+
2024-08-11 00:28:53 | Model                | Version | Status |
2024-08-11 00:28:53 +----------------------+---------+--------+
2024-08-11 00:28:53 | densenet_onnx        | 1       | READY  |
2024-08-11 00:28:53 | inception_graphdef   | 1       | READY  |
2024-08-11 00:28:53 | simple               | 1       | READY  |
2024-08-11 00:28:53 | simple_dyna_sequence | 1       | READY  |
2024-08-11 00:28:53 | simple_identity      | 1       | READY  |
2024-08-11 00:28:53 | simple_int8          | 1       | READY  |
2024-08-11 00:28:53 | simple_sequence      | 1       | READY  |
2024-08-11 00:28:53 | simple_string        | 1       | READY  |
2024-08-11 00:28:53 +----------------------+---------+--------+
```

후에 저 densenet_onnx 를 직접 사용해 테스트 할 것이다.

먼저 ubuntu 환경 내에서 해당 레포를 클론하고, docs/examples 폴더로 이동해 fetch_xxx.sh 파일을 실행해 파일들을 모두 model_repository 폴더에 불러왔다.

처음 docker에 올릴 때 모델을 지정하면 바꿀 수 없기 때문에 거듭해서 server 를 Launch해야 하고, 버전을 따로 잘 적어야 하겠다.

### Docker 컨테이너 생성

```
docker run --gpus=1 -d --name odlm-triton-server -p8000:8000 -p8001:8001 -p8002:8002 -v/home/dohyung/server/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:22.12-py3 tritonserver --model-repository=/models
```

- `-d` : 백그라운드 실행으로, 계속 컨테이너가 존재하도록 해준다.

- `--name odlm-triton-server` : 컨테이너 이름을 지을 수 있다.

- `-v/home/dohyung/server/docs/examples/model_repository:/models` : ubuntu 환경 내에서 모델 저장소로 쓸 폴더를 지정하고, 이를 models 라는 name으로 정하겠다는 의미인 듯 하다. 이 부분에서 잘못 지정해서 한참을 헤매 모델이 없는 컨테이너가 계속 올라갔었다.

- `-py3`, `-py3-sdk` : py3는 일반적인 버전, sdk는 클라이언트 버전이다. 이외에도 `min : 사용자 커스텀 버전` 등이 있다.

- `--model-repository` : 없으면 안되는 모델 저장소를 지정하는 option

- 포트 별로 역할이 다르다. 8000 번 포트는 http 담당으로, 여기로 request를 보낸다.

### Docker Client 컨테이너 생성

```
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:22.12-py3-sdk
```

클라이언트 이미지이다. 단순히 브라우저에서 요청하면 되는 줄 알았는데, 아닌 것도 같아서 더 공부해봐야겠다.

- `it`는 프로그램 내에서 I/O를 수행한다는 의미.
- `--rm`은 일회용으로 쓰고 지우겠다는 의미.
- `--net=host` 는 server랑 같은(?) 영역(?)에서 쉽게 쉽게 접근(???) 한다는 의미.

### 입/출력

```
$ /workspace/install/bin/image_client -m densenet_onnx -c 3 -s INCEPTION /workspace/images/mug.jpg
Request 0, batch size 1
Image '/workspace/images/mug.jpg':
    15.346230 (504) = COFFEE MUG
    13.224326 (968) = CUP
    10.422965 (505) = COFFEEPOT
```

NVIDIA 머그컵 이미지를 제출하면 커피 머그, 컵, 커피팟 으로 결과를 제출하는 것을 볼 수 있다.

---

#### 다음은 HuggingFace의 llama 를 올려서 직접 어플에서 테스트 해보려 한다!!

[triton llama tutorial](https://github.com/triton-inference-server/tutorials/blob/main/Popular_Models_Guide/Llama2/trtllm_guide.md)
