# Kserve

이름이 바뀌어 Kserve가 되었고, Kubeflow에서 독립해 독자적인 모델서빙 오픈소스가 되었다.

[Kubeflow KFserving 그 존재의 이유](https://devocean.sk.com/blog/techBoardDetail.do?ID=163645)

[Kserve Documentation](https://kserve.github.io/website/master/get_started/)

### Create a NameSpace

```r
kubectl create namespace kserve-test
```

### Create an Inference Service

new Schema 이다.

```r
kubectl apply -n kserve-test -f - <<EOF
apiVersion: "serving.kserve.io/v1beta1"
kind: "InferenceService"
metadata:
  name: "sklearn-iris"
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: "gs://kfserving-examples/models/sklearn/1.0/model"
EOF
```

### Perform Inference

```r
cat <<EOF > "./iris-input.json"
{
  "instances": [
    [6.8,  2.8,  4.8,  1.4],
    [6.0,  3.4,  4.5,  1.6]
  ]
}
EOF
```

추론하는 4가지 방법이 있는데

- 실제 DNS 사용
- Magic DNS 사용 (ingressgateway의 External-ip 를 확인해 넣어주기)
- Ingressgateway + Host Header
- Local Cluster gateway

상황 봐서 사용해야 할 듯 하다.
