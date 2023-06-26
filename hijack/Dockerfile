FROM horovod/horovod

RUN apt-get update -y && apt-get install -y vim cmake git
RUN pip install tqdm==4.62.3 scipy==1.6.3 datasets==1.12.1 transformers==4.10.3 grpcio==1.40.0 grpcio-tools==1.40.0 &&\
  pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip uninstall -y horovod tensorflow && pip install --no-cache-dir tensorflow-gpu==1.15.4 && horovod==0.22.1

WORKDIR /