# Kserve

이름이 바뀌어 Kserve가 되었고, Kubeflow에서 독립해 독자적인 모델서빙 오픈소스가 되었다.

[Kubeflow KFserving 그 존재의 이유](https://devocean.sk.com/blog/techBoardDetail.do?ID=163645)

[Kserve Documentation](https://kserve.github.io/website/master/get_started/)

### Create a NameSpace

```r
# kubeflow v1.8 - kserve v0.11.2

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
- **Ingressgateway + Host Header**
- Local Cluster gateway

상황 봐서 사용해야 할 듯 하다.

> 로컬 환경에서는 3번째 방법이 맞는 듯 하다.

### Trouble Shooting

진짜 말도 안된다.

```r

CLUSTER IP : 10.107.3.169
HOST : sklearn-iris.kserve-test.svc.cluster.local

SESSION : MTcyNjAxMTE1OXxOd3dBTkVGS05FRllXVWhWVmxaS1ExUXpTbE5HVDFSYU5FaEpURUZITkZKQ1ZFZEROekpDU1RkVlJ6VkZWMGRKV2s1WFZEY3lNa0U9fNOgH7xOjYaedrJR50gDDjrqVj_TPqYtmwiFaTKGJnBq


# 직접 클러스터에 매다 꽂아서 성공한 요청
curl -v -H "Host: sklearn-iris.kserve-test.svc.cluster.local" -H "Content-Type: application/json" -H "Cookie: authservice_session=MTcyNjAxMTE1OXxOd3dBTkVGS05FRllXVWhWVmxaS1ExUXpTbE5HVDFSYU5FaEpURUZITkZKQ1ZFZEROekpDU1RkVlJ6VkZWMGRKV2s1WFZEY3lNa0U9fNOgH7xOjYaedrJR50gDDjrqVj_TPqYtmwiFaTKGJnBq" http://10.0.0.147:8080/v1/models/sklearn-iris:predict -d @./iris-input.json



# Ingress-gateway 를 NodePort로 설정하고, Tutorial을 따라한다.
# 301 Moved Permanently 가 뜬다.
curl -v -H "Host: sklearn-iris.kserve-test.svc.cluster.local" -H "Content-Type: application/json" -H "Cookie: authservice_session=MTcyNjAxMTE1OXxOd3dBTkVGS05FRllXVWhWVmxaS1ExUXpTbE5HVDFSYU5FaEpURUZITkZKQ1ZFZEROekpDU1RkVlJ6VkZWMGRKV2s1WFZEY3lNa0U9fNOgH7xOjYaedrJR50gDDjrqVj_TPqYtmwiFaTKGJnBq" "http://172.26.42.180:30749/v1/models/sklearn-iris:predict" -d @./iris-input.json --insecure
```

분명 Ingressgateway 에서 요청을 받으면 해당 클러스터로 쏴줘야하는데,

`VirtualService`, `Gateway`등의 설정에 대한 이해가 너무 부족하다.

어떤게 가능하고, 불가능한지. 어떻게 해야 하는지에 대한 방향성 없이 막연하게 이러면 되지 않을까를 남발하고 있다.

몇가지 알아낸 포인트들은 다음과 같다.

- 인증이 필요하다면, 포트포워딩으로 접속한 `CentralBoard`에서 개발자 도구를 켜서 담겨진 쿠키를 훔쳐볼 수 있다. 해당 쿠키를 요청에서 헤더에 authservice_session으로 담아주면 된다.

- 인증이 번거롭다면, 필요한 서비스의 `deployment.apps` 를 수정해 APP_SECURE_COOKIES 를 "False" 로 지정해서 생략할 수 있다.

-
