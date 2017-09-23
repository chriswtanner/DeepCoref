INSTALL
-------
pip install virtualenv
mkdir venv
source venv/bin/activate

brew install python3
pip3 install --user http://download.pytorch.org/whl/torch-0.2.0.post3-cp36-cp36m-macosx_10_7_x86_64.whl 
pip3 install torchvision

./pip install -U numpy scipy scikit-learn tensorflow keras matplotlib

# cd to the src/
git clone https://github.com/harveyslash/Facial-Similarity-with-Siamese-Networks-in-Pytorch.git