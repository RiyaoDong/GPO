wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip -d data/
mv data/tiny-imagenet-200 data/tiny_imagenet
mkdir data/tiny_imagenet/valset
python pro_tiny_image.py

