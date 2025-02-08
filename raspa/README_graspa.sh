conda create -n graspa python==3.10
# comp/gcc/12.3.0
# comp/cmake/3.31.1
lib/cuda/12.4
comp/nvhpc/nvhpc-23.11
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117
wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcu117.zip
ln -s libnvrtc-builtins-7237cb5d.so.11.7  libnvrtc-builtins.so.11.7


look at: https://github.com/snurr-group/gRASPA/tree/main/Cluster-Setup/NERSC
