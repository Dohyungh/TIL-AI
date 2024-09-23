# Trouble Shooting

SSAFY에서 제공하는 GPU 서버에서 sudo 명령어를 쓸 수 없어 직접적인 Kubeflow 설치가 불가능해진 이후, 팀에 주어진 EC2 서버에 Kubeflow를 설치하고, GPU 서버없이 껍데기만 운용하는 방식을 시도한다.

### Kubeadm init

kubernetes 내부의 api 시스템이 제대로 작동하지 않아서, cert-manager 와 local-path-storage pod 등이 제대로 Running 상태가 되지 않는 오류가 있어 해결 중이다.

![alt text](image.png)

#### 해결!

클러스터 내부 네트워크 문제인 것 같다는 생각은 들었는데, 역시 맞았다. 다음의 명령어가 문제였다.

```r

sudo kubeadm reset

#만약, unmount dir 어쩌구에서 한참 머물고 있다면,
# sudo systemctl stop containerd
# sudo kubeadm reset
# sudo systemctl start containerd
# sudo kubeadm init
# 위와 같이, containerd를 잠시 꺼놓고 reset한다음 다시 키고 kubeadm init 하면 된다.

# 기본 설정
sudo sysctl -p
sudo swapoff -a

# kubelet 까지만 잘 켜지는지 확인
sudo kubeadm init phase kubelet-start

# cilium 을 쓰냐, calico 혹은 Flannel을 쓰냐에 따라 --pod-network-cidr의 주소가 달랐다.

# before
sudo kubeadm init --pod-network-cidr=10.217.0.0/16

# after (cilium)
sudo kubeadm init --pod-network-cidr=10.1.1.0/24

## !!수정
## 결국 저것도 아니었다. kind cluster 로 생성할 때 지정해준다. kubeflow 1.9ver 기준.
Networking.PodSubnet     = "10.244.0.0/16"
Networking.ServiceSubnet = "10.96.0.0/12"

cat <<EOF | kind create cluster --name=kubeflow --config=-
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  image: kindest/node:v1.29.4
  kubeadmConfigPatches:
  - |
    kind: ClusterConfiguration
    apiServer:
      extraArgs:
        "service-account-issuer": "kubernetes.default.svc"
        "service-account-signing-key-file": "/etc/kubernetes/pki/sa.key"
networking:
  disableDefaultCNI: true
EOF


```

### cilium

```r
cilium status --wait
```

위 명령어로 상태를 지켜보는데 계속해서 operator가 error 이길래 살펴보았다.

![alt text](image-1.png)

[naver cloud platform](https://guide-gov.ncloud-docs.com/docs/k8s-troubleshoot-common)

![alt text](image-2.png)

control-plane 하나이기 때문에 의도된 바라고 한다.

### dial tcp connection refused

```r
Error from server (InternalError): error when creating "STDIN": Internal error occurred: failed calling webhook "clusterservingruntime.kserve-webhook-server.validator": failed to call webhook: Post "https://kserve-webhook-server-service.kubeflow.svc:443/validate-serving-kserve-io-v1alpha1-clusterservingruntime?timeout=10s": dial tcp 10.96.115.86:443: connect: connection refused
```

위의 에러가 설치 과정에서 지속적으로 발생했다.

#### 해결!

Nginx 를 끄고, Cert-manager 를 적용해줬다.

```r
kubectl wait --for=condition=ready pod -l 'app in (cert-manager,webhook)' --timeout=180s -n cert-manager
kubectl wait --for=jsonpath='{.subsets[0].addresses[0].targetRef.kind}'=Pod endpoints -l 'app in (cert-manager,webhook)' --timeout=180s -n cert-manager
```

아마 443 포트 문제였던게 아닌가 싶다.

### CPU Resource 문제

```r
Allocated resources:
  (Total limits may be over 100 percent, i.e., overcommitted.)
  Resource           Requests      Limits
  --------           --------      ------
  cpu                3980m (99%)   34300m (857%)
  memory             6994Mi (43%)  18248Mi (114%)
  ephemeral-storage  0 (0%)        0 (0%)
  hugepages-1Gi      0 (0%)        0 (0%)
  hugepages-2Mi      0 (0%)        0 (0%)
```

안그래도 manifests 에 적혀있기를

```r
32 GB of RAM recommended
16 CPU cores recommended
```

가 있어서 불안했는데, 역시나 CPU를 전부 점유하고 있어서

```r
auth                 dex-7b97f48486-2x7gz                                     1/1     Running            0               25m
cert-manager         cert-manager-7ddd8cdb9f-96lbl                            1/1     Running            0               25m
cert-manager         cert-manager-cainjector-57cd76c845-982bt                 1/1     Running            0               25m
cert-manager         cert-manager-webhook-cf8f9f895-mpzr5                     1/1     Running            0               25m
istio-system         cluster-local-gateway-595b55bdb4-g4gd4                   1/1     Running            0               25m
istio-system         istio-ingressgateway-66c5cb4897-vbb27                    1/1     Running            0               25m
istio-system         istiod-66f745bccb-nsknq                                  1/1     Running            0               25m
knative-eventing     eventing-controller-5bfd896747-b6gnw                     1/1     Running            0               25m
knative-eventing     eventing-webhook-5b7d5d858-bjnv2                         1/1     Running            0               25m
knative-serving      activator-54c999bcbc-dfnf8                               0/2     Pending            0               25m
knative-serving      autoscaler-776bbf9496-kl5p7                              0/2     Pending            0               25m
knative-serving      controller-854dcfd65b-cmzwm                              0/2     Pending            0               25m
knative-serving      net-istio-controller-79d44bc46d-f9p92                    0/2     Pending            0               25m
knative-serving      net-istio-webhook-ff94788ff-ccgrr                        0/2     Pending            0               25m
knative-serving      webhook-56bccb4b4f-ftw4t                                 0/2     Pending            0               25m
kube-system          cilium-6kbtv                                             1/1     Running            0               27m
kube-system          cilium-envoy-z5f4v                                       1/1     Running            0               27m
kube-system          cilium-operator-8547744bd7-5mdjx                         0/1     Pending            0               27m
kube-system          cilium-operator-8547744bd7-f27lz                         1/1     Running            0               27m
kube-system          coredns-76f75df574-44cd6                                 1/1     Running            0               27m
kube-system          coredns-76f75df574-6g5lh                                 1/1     Running            0               27m
kube-system          etcd-kubeflow-control-plane                              1/1     Running            0               28m
kube-system          kube-apiserver-kubeflow-control-plane                    1/1     Running            0               28m
kube-system          kube-controller-manager-kubeflow-control-plane           1/1     Running            0               28m
kube-system          kube-proxy-t74bg                                         1/1     Running            0               27m
kube-system          kube-scheduler-kubeflow-control-plane                    1/1     Running            0               28m
kubeflow             admission-webhook-deployment-6d796b475c-xbcn7            1/1     Running            0               25m
kubeflow             cache-server-69f48c65dd-9c4fk                            2/2     Running            1 (8m41s ago)   25m
kubeflow             centraldashboard-7499dffcf8-qrkmf                        2/2     Running            0               25m
kubeflow             jupyter-web-app-deployment-58c6ddc5c9-dfmhc              2/2     Running            0               25m
kubeflow             katib-controller-6c5684bf6b-rx68j                        1/1     Running            0               25m
kubeflow             katib-db-manager-7866998cf9-9f55f                        1/1     Running            1 (23m ago)     25m
kubeflow             katib-mysql-76b79df5b5-d2246                             1/1     Running            0               25m
kubeflow             katib-ui-657777bbff-v6rd5                                2/2     Running            0               25m
kubeflow             kserve-controller-manager-5ddf84f8d4-gmr2w               2/2     Running            0               25m
kubeflow             kserve-models-web-app-8485988f76-86ltj                   2/2     Running            0               25m
kubeflow             kubeflow-pipelines-profile-controller-5dcf75b69f-rbrxr   1/1     Running            0               25m
kubeflow             metacontroller-0                                         0/1     Pending            0               25m
kubeflow             metadata-envoy-deployment-9c7db86d8-tq4h4                1/1     Running            0               25m
kubeflow             metadata-grpc-deployment-d94cc8676-rlsll                 1/2     CrashLoopBackOff   9 (95s ago)     25m
kubeflow             metadata-writer-d9bc4bb89-gfc5c                          1/2     CrashLoopBackOff   6 (4m58s ago)   25m
kubeflow             minio-5dc6ff5b96-xkrjc                                   2/2     Running            0               25m
kubeflow             ml-pipeline-5846b5b56d-qh8h2                             1/2     Running            9 (55s ago)     25m
kubeflow             ml-pipeline-persistenceagent-7655ddbcfb-rqvxm            2/2     Running            0               25m
kubeflow             ml-pipeline-scheduledworkflow-658b675548-blcfk           2/2     Running            0               25m
kubeflow             ml-pipeline-ui-5c66bf88b5-bdvtv                          2/2     Running            0               25m
kubeflow             ml-pipeline-viewer-crd-c4d866c85-mzxp7                   2/2     Running            1 (21m ago)     25m
kubeflow             ml-pipeline-visualizationserver-7c678699d-c99ns          2/2     Running            0               25m
kubeflow             mysql-5b446b5744-4jzsp                                   0/2     Pending            0               25m
kubeflow             notebook-controller-deployment-5458bf988b-cmdz6          0/2     Pending            0               25m
kubeflow             profiles-deployment-79f5cf977d-p7j89                     0/3     Pending            0               25m
kubeflow             pvcviewer-controller-manager-7979499b66-f24mp            0/3     Pending            0               25m
kubeflow             tensorboard-controller-deployment-78f5598f4b-vpddd       0/3     Pending            0               25m
kubeflow             tensorboards-web-app-deployment-6dc87f944-z4gmx          0/2     Pending            0               25m
kubeflow             training-operator-7b99976c6-sdqvr                        1/1     Running            0               25m
kubeflow             volumes-web-app-deployment-db79f546d-48xnx               0/2     Pending            0               25m
kubeflow             workflow-controller-68d76bcb8b-8pnxj                     2/2     Running            1 (18m ago)     25m
local-path-storage   local-path-provisioner-765484d458-7rgrq                  1/1     Running            0               24m
oauth2-proxy         oauth2-proxy-86d8c97455-mvcdw                            1/1     Running            0               25m
oauth2-proxy         oauth2-proxy-86d8c97455-rzwxd                            1/1     Running            0               25m
```

대부분의 pod 들이 pending 상태이다.

버전을 낮춰야 한다. 즉, kubernetes 설치부터 다시한다..!
