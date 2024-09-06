FROM kubeflownotebookswg/jupyter-pytorch-cuda-full:v1.8.0
USER root 

# Nvidia GPG Public Key 교체
RUN rm -f /etc/apt/sources.list.d/cuda.list \
  && rm -f /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y --no-install-recommends \
	  sudo \
	  apt-utils \
  && usermod -aG sudo ${NB_USER} \
  && echo ${NB_USER}:${NB_USER} | chpasswd \
  && echo "${NB_USER} ALL=(root) NOPASSWD:SETENV: /init" >> /etc/sudoers

# install - requirements.txt
# COPY requirements.txt requirements.txt
# RUN pip3 install -r requirements.txt
EXPOSE 8888
USER jovyan

ENV NB_USER $NB_USER
ENV NB_UID=1000
ENV HOME /home/$NB_USER
ENV NB_PREFIX /

CMD ["sh","-c", "jupyter notebook --notebook-dir=/home/jovyan --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=${NB_PREFIX}"]