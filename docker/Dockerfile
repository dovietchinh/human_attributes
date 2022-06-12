FROM nvidia/cuda:11.0-devel-ubuntu20.04
RUN mkdir -p workspace
RUN rm /etc/apt/sources.list.d/cuda.list
RUN /bin/bash -c "apt-get update"
RUN /bin/bash -c "apt-get install python3-pip -y"
RUN pip3 install numpy 
RUN pip3 install matplotlib
RUN pip3 install pandas
RUN pip3 install tqdm
RUN pip3 install PyYAML
RUN pip3 install opencv-python 
RUN pip3 install torch==1.10
RUN pip3 install dvc
RUN apt-get install git
RUN git config --global --add safe.directory /workspace
WORKDIR /workspace

