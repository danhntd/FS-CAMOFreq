# Conda environment for iFS-RCNN
conda create --name fsdet python=3.9
conda activate fsdet

# PyTorch
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# Detectron2
git clone https://github.com/facebookresearch/detectron2.git
git checkout v0.2.1
pip install -e detectron2

# Pip libraries
pip install opencv-python-headless scikit-learn
pip install pillow==9.5
pip install numpy==1.23.0