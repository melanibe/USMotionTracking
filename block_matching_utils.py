import numpy as np
import parmap

''' 
Mélanie Bernhardt - ETH Zurich
CLUST Challenge
'''


def find_search_pixel(c1, c2, width=100):
    '''
    Defines the grid search for the block matching
    '''
    nb_points = width/2+1
    xax = np.linspace(c1-width/2, c1+width/2, nb_points, dtype='int')
    yax = np.linspace(c2-width/2, c2+width/2, nb_points, dtype='int')
    return np.meshgrid(xax, yax)


def find_template_pixel(c1, c2, width=60):
    ''' Find the pixels coordinates for the
    template centered around c1, c2
    '''
    w = width//2
    xax = np.arange(int(c1-1)-w, int(c1-1)+w+1, step=1)
    yax = np.arange(int(c2-1)-w, int(c2-1)+w+1, step=1)
    # all the x,y in the template centered around c1, c2
    return np.meshgrid(xax, yax)

def get_NCC(i, j, im1, im2, width, yv, xv):
    '''Returns the NCC between 2 images
    '''
    tmp_x, tmp_y = find_template_pixel(i, j, width)
    try:
        x1 = np.ravel(im1[np.ravel(yv), np.ravel(xv)])
        x2 = np.ravel(im2[np.ravel(tmp_y), np.ravel(tmp_x)])
        if np.percentile(x2, 0.4) == 0:
            # if 40% is black it means you are on the border
            # i.e. bad choice
            return 0
        x1 = x1 - np.mean(x1)
        x2 = x2 - np.mean(x2)
        num = np.sum(x1*x2)
        denom = np.sqrt(np.sum(x1**2)*np.sum(x2**2))
        if denom == 0:
            return 0
        else:
            return num/denom
    except IndexError:
        raise
        return 0  

def NCC_best_template_search(c1, c2, im1, im2, width=60, c1_init=None, c2_init=None, search_w = 100):
    ''' Finds the best center according to block matching search
    '''
    print(c1,c2)
    searchx, searchy = find_search_pixel(c1, c2, search_w)
    print(np.min(c1), np.max(c1))
    print(np.min(c2), np.max(c2))
    if c1_init is None:
        xv, yv = find_template_pixel(c1, c2, width)
    else:
        xv, yv = find_template_pixel(c1_init, c2_init, width)
    NCC_all = parmap.starmap(get_NCC, zip(np.ravel(searchx), np.ravel(searchy)), im1, im2, width, yv, xv, pm_parallel=True)
    maxNCC = np.max(NCC_all)
    idx = np.argmax(NCC_all)
    best_c1, best_c2 = np.ravel(searchx)[idx], np.ravel(searchy)[idx]
    return best_c1, best_c2, maxNCC
