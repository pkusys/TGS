# Benchmark Megatron-LM

## Requirements
```
# torch 1.8.0 and CUDA 11.1
pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

pip3 install ninja

# Install Megatron (version==v3.0)
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
echo 'export PYTHONPATH=$PYTHONPATH:~/efs/Megatron-LM' >> ~/.bashrc   # use your own path
source ~/.bashrc

# Install Apex (version==22.04-dev)
git clone -b 22.04-dev https://github.com/NVIDIA/apex
cd apex
# Comment out the raised RuntimeError in setup.py if you get errors running the following command.
pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
For convenience, you can pull our docker image with the above dependencies installed:
```
docker pull goldensea/megatron:v2
```


## Instructions
### Single Node with two GPUs
```
# GPT
python benchmark_gpt_bert.py --nproc_per_node 2 --suite gpt.tmp
```