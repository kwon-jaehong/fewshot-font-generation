FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu18.04
RUN apt update
RUN apt-get install -y python3.7 python3.7-dev wget curl python3.7-distutils git apt-transport-https

RUN git clone https://github.com/kwon-jaehong/fewshot-font-generation.git
WORKDIR /fewshot-font-generation

RUN python3.7 get-pip.py
RUN apt install -y libsm6 libfontconfig1 libxrender1 libxtst6 libglib2.0-0 libgl1-mesa-glx gcc 
RUN pip install --upgrade pip
RUN pip install opencv-python pillow tqdm
# RUN pip install -r ./requirements.txt
# RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
# RUN pip install fontTools sconf tqdm scipy tensorboard tensorboardX