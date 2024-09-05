# jupyter-nb-image 생성

친절한 Tutorial과 설명이 [Kubeflow - Custom Images](https://www.kubeflow.org/docs/components/notebooks/container-images/)에 있음

Image는 다음과 같은 사항을 만족해야 함

- expose an HTTP interface on port 8888:
  - kubeflow sets an environment variable NB_PREFIX at runtime with the URL path we expect the container be listening under
  - kubeflow uses IFrames, so ensure your application sets Access-Control-Allow-Origin: \* in HTTP response headers
- run as a user called jovyan:
  - the home directory of jovyan should be /home/jovyan
  - the UID of jovyan should be 1000
- start successfully with an empty PVC mounted at /home/jovyan:
  - kubeflow mounts a PVC at /home/jovyan to keep state across Pod restarts

notebook terminal에서 sudo가 안먹어서 새로운 이미지 만들어서 도커 띄워야함

```r

docker build --tag jupyter_nb_pytorch:1.5.0 .

```
