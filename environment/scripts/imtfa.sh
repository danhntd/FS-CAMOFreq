# Conda environment for iFS-RCNN
conda create --name imtfa python=3.9
conda activate imtfa

# PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

pip install cython
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install opencv-python-headless scikit-learn
pip install pillow==9.5

rm -rf build/ **/*.so

apt install software-properties-common
add-apt-repository ppa:ubuntu-toolchain-r/test
apt update && install gcc-12 g++-12 -y
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 12 --slave /usr/bin/g++ g++ /usr/bin/g++-12

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST=6.0,7.0,8.0,8.6
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
pip install -e .

pip install numpy==1.23.0