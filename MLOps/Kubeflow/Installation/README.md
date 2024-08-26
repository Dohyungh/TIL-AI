# Installation

[Kubeflow Essentials](https://www.youtube.com/watch?v=qoqLtrdAXQg&t=177s) 강의를 보고 작성됨

ubuntu 20.04
gpu : rtx 3070

## 1. install cudnn

https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html

```r
sudo apt-get -y update
sudo apt-get -y remove --purge '^nvidia-.\*'
sudo apt-get -y remove --purge 'cuda-.\*'
```

[nvidia cuda toolkit, V.11.8 설치](https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

```r
nvcc -V
```

---

안 될 시

`~/.bashrc` 에 다음 내용 추가

```r
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
```

```r
source ~/.bashrc
```

---

```r
whereis cuda
mkdir ~/nvidia
cd ~/nvidia
```

cuDNN v9.3.0

https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local

```r
wget https://developer.download.nvidia.com/compute/cudnn/9.3.0/local_installers/cudnn-local-repo-ubuntu2004-9.3.0_1.0-1_amd64.deb
sudo dpkg -i cudnn-local-repo-ubuntu2004-9.3.0_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-ubuntu2004-9.3.0/cudnn-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cudnn
```

## 2. install docker

```r
sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release

## key error 나면 PUB_KEY 자리에 KEY 넣어서
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys <PUB_KEY>

sudo install -m 0755 -d /etc/apt/keyrings

sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
# sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo docker run hello-world
sudo usermod -aG docker $USER && newgrp docker
sudo service docker restart
```

## 3. install nvidia-docker

```r
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get -y update
sudo apt-get -y install nvidia-docker2
sudo systemctl restart docker
sudo docker run --runtime nvidia nvidia/cuda:11.8.0-base-ubuntu20.04 /usr/bin/nvidia-smi


sudo bash -c 'cat <<EOF > /etc/docker/daemon.json
{
    "exec-opts": ["native.cgroupdriver=systemd"],
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "100m"
    },
    "data-root": "/mnt/storage/docker_data",
    "storage-driver": "overlay2",
    "default-runtime" : "nvidia",
    "runtimes" : {
        "nvidia" : {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs" : []
        }
    }
}
EOF'
sudo systemctl restart docker
```

## 4. install k8s

```r
sudo swapoff -a
sudo sed -i '/ swap / s/^\(.*\)$/#\1/g' /etc/fstab

sudo apt-get install -y iptables arptables ebtables
sudo apt-get update && sudo apt-get install -y apt-transport-https curl
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

## 수작업 nano
cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list deb https://apt.kubernetes.io/ kubernetes-xenial main EOF

sudo apt-get update
# sudo apt-get install -y kubelet=1.21.10-00 kubeadm=1.21.10-00 kubectl=1.21.10-00 --allow-downgrades --allow-change-held-packages

# 더 이상 install안 됨. 도커 지원 안함. 1.24 kubernetes 써야함

curl -fsSL https://pkgs.k8s.io/core:/stable:/v1.24/deb/Release.key | sudo gpg --dearmor -o /etc/apt/keyrings/kubernetes-apt-keyring.gpg
echo "deb [signed-by=/etc/apt/keyrings/kubernetes-apt-keyring.gpg] https://pkgs.k8s.io/core:/stable:/v1.24/deb/ /" | sudo tee /etc/apt/sources.list.d/kubernetes.list

sudo apt-mark hold kubelet kubeadm kubectl
kubeadm version
kubelet --version
kubectl version --client
```
