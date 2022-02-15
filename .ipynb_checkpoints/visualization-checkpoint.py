import numpy as np
import matplotlib.pyplot as plt

def scatter_3d(X, labs=None, title=None):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    x, y, z = zip(*X)
    
    if labs is None:
        ax.scatter(x, y, z)
    else:
        ax.scatter(x, y, z, c=labs, cmap='PiYG')
    
    if title is not None:
        ax.set_title(title)
        
    plt.show()
    
def plot_2d(X, title=None):
    fig, ax = plt.subplots()
    
    x, y = zip(*X)
    
    ax.plot(x, y)
    
    if title is not None:
        ax.set_title(title)
        
    plt.show()