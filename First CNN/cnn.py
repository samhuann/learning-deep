import mnist
from conv import Conv3x3
from loader import load_mnist_images, load_mnist_labels
from maxpool import MaxPool2

train_images = load_mnist_images("mnist_data/train-images.idx3-ubyte") #load training images
train_labels = load_mnist_labels("mnist_data/train-labels.idx1-ubyte") #load training labels

conv = Conv3x3(8) #use 8 filters
pool = MaxPool2()

output = conv.forward(train_images[0]) #pass first image thru layer=
output = pool.forward(output)
print(output.shape)

