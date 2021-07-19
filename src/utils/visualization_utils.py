import os
import numpy as np
import matplotlib.pyplot as plt

def show_decoded_im(imout, savedir, epoch=0):
    """
    Args:
        im: output sample of 3 channels (C x H x W)
    """
    imout = imout.transpose((1,2,0))
    for i in range(3):
        plt.subplot(2,2,i+1)
        plt.imshow(imout[:,:,i])
        plt.axis('off')
    plt.subplot(224)
    plt.imshow(imout)
    plt.axis('off')
    plt.savefig(os.path.join(savedir,'adaptor_%d.png'%(epoch)))
    #plt.show()
