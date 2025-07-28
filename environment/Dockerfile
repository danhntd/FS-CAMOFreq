FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# update the package list
RUN apt-get update && apt-get install -y

# Install essential packages
RUN apt-get install -y vim htop wget curl net-tools git tmux locales screen nano iputils-ping

# Download and install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# Install C++ build tools
RUN apt-get install -y build-essential software-properties-common \
    && add-apt-repository ppa:ubuntu-toolchain-r/test \
    && apt update && apt install gcc-12 g++-12 -y \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12 --slave /usr/bin/g++ g++ /usr/bin/g++-12

# Set up locale
RUN locale-gen en_US.UTF-8 && \
    update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# Set environment variables
ENV PATH=/opt/conda/bin:$PATH
ENV TERM=xterm-256color
ENV CC=/usr/bin/gcc-12
ENV CXX=/usr/bin/g++-12

# Init conda environments
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda create --name dcfs python=3.9 -y && \
    conda create --name fsdet python=3.9 -y && \
    conda create --name imtfa python=3.9 -y

# Install iMTFA environment
RUN /opt/conda/bin/conda run -n imtfa pip install --upgrade pip --root-user-action=ignore
RUN /opt/conda/bin/conda run -n imtfa pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --root-user-action=ignore
RUN /opt/conda/bin/conda run -n imtfa pip install cython --root-user-action=ignore && \
    /opt/conda/bin/conda run -n imtfa pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' --root-user-action=ignore && \
    /opt/conda/bin/conda run -n imtfa pip install opencv-python-headless scikit-learn --root-user-action=ignore && \
    /opt/conda/bin/conda run -n imtfa pip install pillow==9.5 numpy==1.23.0 --root-user-action=ignore

# Install fsdet environment
RUN /opt/conda/bin/conda run -n fsdet pip install --upgrade pip --root-user-action=ignore
RUN /opt/conda/bin/conda run -n fsdet pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --root-user-action=ignore
RUN /opt/conda/bin/conda run -n fsdet pip install opencv-python-headless scikit-learn --root-user-action=ignore && \
    /opt/conda/bin/conda run -n fsdet pip install pillow==9.5 numpy==1.23.0 --root-user-action=ignore

# Copy configuration files
COPY ./config/.tmux.conf /root/.tmux.conf
COPY ./config/.bashrc /root/.bashrc

# Entrypoint
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]