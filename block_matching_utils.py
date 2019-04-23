import numpy as np

# Find pixels around a given center
def find_search_pixel(c1, c2, width=100):
    xax = np.linspace(c1-width/2,c1+width/2,51,dtype='int')
    yax = np.linspace(c2-width/2,c2+width/2,51,dtype='int')
    #xax = np.arange(int(c1)-25, int(c1)+26, step=1)
    #yax = np.arange(int(c2)-25, int(c2)+26, step=1)
    return np.meshgrid(xax, yax) # all the x,y in the template centered around c1, c2

def find_template_pixel(c1, c2, width=100):
    w = width//2
    xax = np.arange(int(c1-1)-w, int(c1-1)+w+1, step=1)
    yax = np.arange(int(c2-1)-w, int(c2-1)+w+1, step=1)
    return np.meshgrid(xax, yax) # all the x,y in the template centered around c1, c2