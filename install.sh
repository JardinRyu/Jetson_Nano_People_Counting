echo "** Set up PATH"
sudo jetson_clocks
sudo apt update

cd install_scripts/
./install_basics.sh
source ~/.bashrc

echo "** Install dependencies for cv2 "
sudo apt update

sudo apt install -y build-essential make cmake cmake-curses-gui \
                      git g++ pkg-config curl libfreetype6-dev \
                      libcanberra-gtk-module libcanberra-gtk3-module \
                      python3-dev python3-pip
sudo pip3 install -U pip==20.2.1 Cython testresources setuptools

./install_protobuf-3.8.0.sh

sudo pip3 install numpy==1.16.1 matplotlib==3.2.2

echo "** Install Tensorflow"
sudo apt install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev \
                      zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo pip3 install -U numpy==1.16.1 future==0.18.2 mock==3.0.5 h5py==2.10.0 \
                       keras_preprocessing==1.1.1 keras_applications==1.0.8 \
                       gast==0.2.2 futures pybind11
sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 'tensorflow<2'

echo "** Install Tensorflow successfully"

#echo "Set up Detection"
#./ssd/install.sh
#./ssd/build_engines.sh

sudo pip3 install filterpy
sudo apt-get install python3-sklearn
sudo apt-get install llvm-7
sudo LLVM_CONFIG=/usr/bin/llvm-config-7 pip3 install llvmlite==0.32.0
sudo LLVM_CONFIG=/usr/bin/llvm-config-7 pip3 install numba==0.43.0