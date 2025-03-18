import mnist
import numpy as np
from conv import Conv3x3
from loader import load_mnist_images, load_mnist_labels, load_test_data
from maxpool import MaxPool2
from softmax import Softmax

#train_images = load_mnist_images("mnist_data/train-images.idx3-ubyte") #load training images
#train_labels = load_mnist_labels("mnist_data/train-labels.idx1-ubyte") #load training labels



test_images, test_labels = load_test_data()

conv = Conv3x3(8) #use 8 filters
pool = MaxPool2()
softmax = Softmax(13 * 13 * 8, 10) #take 13x13x8 as input, output one in 10 possibilities

def forward(image, label):
    out = conv.forward((image/255)-0.5) #transform from [0,255] to [-0.5,0.5], normalizing image
    out = pool.forward(out)
    out = softmax.forward(out)
    
    #calculate crossentropy loss and accuracy 
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0 #find index of predicted digit. Return 1 if true 

    return out, loss, acc

print('MNIST CNN initialized!')

loss = 0
num_correct = 0
for i, (im,label) in enumerate(zip(test_images, test_labels)):
    _, l, acc = forward(im, label)
    loss +=1
    num_correct +=acc
    
    if i%100 == 99: #print stats every 100 steps
        print(
            'Step[%d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i+1, loss/100, num_correct)
        )
        loss = 0
        num_correct = 0


