import numpy as np

class Conv3x3:
    #convolution layer using 3x3 filters

    def __init__(self, num_filters):
        self.num_filters = num_filters

        self.filters = np.random.randn(num_filters, 3,3)/9

        #filters is 3x3 array with dimensions (num_filters, 3, 3)
        #divid by 9 to reduce variance of initial values

    def iterate_regions(self, image):
        h, w = image.shape

        for i in range(h-2):
            for j in range(w-2):
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region, i, j
        #slider 3x3

    def forward(self, input):
        #perform forward pass of conv layer using given input
        #returns 3d numpy array with dimensions (h, w, num_filters). input is 2d numpy array

        h, w = input.shape #receive height and width of input

        output = np.zeros((h-2, w-2, self.num_filters)) #create empty output array

        for im_region, i, j in self.iterate_regions(input): #slider 3x3
            output[i,j] = np.sum(im_region * self.filters, axis=(1,2))#multiply, and sum over height and width
        #axis=(1,2) produces a 1d array of length num_filters
        return output
