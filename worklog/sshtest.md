### first ssh

This is the first time to succeed in ssh to my own computer, recorded by this md.

# pytorch image
docker pull pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel
docker run --gpus all -it --network host --name cudaOldie pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime bash