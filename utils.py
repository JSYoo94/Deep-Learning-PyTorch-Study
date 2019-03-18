import sys, os
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from torchvision.utils import make_grid
from torchvision.utils import save_image

def save_images(img, dir, file_name):
    grid_img = torchvision.utils.make_grid(img.cpu(), nrow=10)
    
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    torchvision.utils.save_image(grid_img, os.path.join(dir, file_name))

def show_images(img_list):
    plt.style.use('default')
    
    fig = plt.figure()
    subplot = []
    
    for i in range(len(img_list)):
        subplot.append(fig.add_subplot(1, len(img_list), i+1))
        grid_img = torchvision.utils.make_grid(img_list[i].cpu(), nrow=10)
        subplot[i].imshow(grid_img.detach().numpy().transpose(1, 2, 0))
        
    plt.show()
                                                       
def drawLoss(loss_dict):
    plt.style.use(['ggplot'])
    
    for key, value in loss_dict.items():
        x = np.arange(len(loss_dict[key]))
        plt.plot(x, loss_dict[key], label=key)
    
    plt.xlabel("train step")
    plt.ylabel("loss")
    
    plt.legend(loc="lower right")
    plt.show()