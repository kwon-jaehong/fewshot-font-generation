FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu18.04
ADD . /fewshot_font_generation
WORKDIR /fewshot_font_generation
RUN apt update
RUN apt-get install -y python3.7 python3.7-dev wget curl python3.7-distutils git apt-transport-https
RUN python3.7 get-pip.py
RUN apt install -y libsm6 libfontconfig1 libxrender1 libxtst6 libglib2.0-0 libgl1-mesa-glx gcc 
RUN pip install -r ./requirements.txt