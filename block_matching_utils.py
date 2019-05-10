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


def find_template_pixel(c1, c2, width, max_x, max_y):
    ''' Find the pixels coordinates for the
    template centered around c1, c2
    '''
    w = width//2
    if c1-1-w<0:
        xax = np.arange(0, w+w+2, step=1)
    elif c1+w > max_x:
        xax = np.arange(max_x-(w+w+2), max_x, step=1)
    else:
        xax = np.arange(int(c1-1)-w, int(c1-1)+w+1, step=1)
    if c2-1-w<0:
        yax = np.arange(0, w+w+2, step=1)
    elif c2+w > max_y:
        yax = np.arange(max_y-w+w+2, max_y, step=1)    
    else:
        yax = np.arange(int(c2-1)-w, int(c2-1)+w+1, step=1)
    # all the x,y in the template centered around c1, c2
    return np.meshgrid(xax, yax)


def get_NCC(i, j, im1, im2, width, yv, xv):
    '''Returns the NCC between 2 images
    '''
    a, b = np.nonzero(im1[:, 20:(len(im1)-20)])
    lim_left = b[np.where(a == np.floor(j))][0]+20
    lim_right = b[np.where(a == np.floor(j))][-1]+20
    # detect the edges
    try:
        if ((lim_left > i)
                or (lim_right < i)):
            #print('proposed center outside image')
            #print(i, j)
            #print(b[np.where(a == j)][0], b[np.where(a == j)][-1])
            return -1
    except IndexError:
        return -1
    tmp_x, tmp_y = find_template_pixel(i, j, width, im2.shape[1], im2.shape[0])
    try:
        x1 = np.ravel(im1[np.ravel(yv), np.ravel(xv)])
        x2 = np.ravel(im2[np.ravel(tmp_y), np.ravel(tmp_x)])
        x1 = x1 - np.mean(x1)
        x2 = x2 - np.mean(x2)
        num = np.sum(x1*x2)
        denom = np.sqrt(np.sum(x1**2)*np.sum(x2**2))
        if denom == 0:
            return -1
        else:
            return num/denom
    except IndexError:
        return -1


def NCC_best_template_search(c1, c2, im1, im2, width=60, c1_init=None, c2_init=None, search_w=100):
    ''' Finds the best center according to block matching search
    '''
    searchx, searchy = find_search_pixel(c1, c2, search_w)
    if c1_init is None:
        xv, yv = find_template_pixel(c1, c2, width, im1.shape[1], im1.shape[0])
    else:
        xv, yv = find_template_pixel(c1_init, c2_init, width, im1.shape[1], im1.shape[0])
    NCC_all = parmap.starmap(get_NCC, zip(np.ravel(searchx), np.ravel(
        searchy)), im1, im2, width, yv, xv, pm_parallel=True)
    maxNCC = np.max(NCC_all)
    if np.sum(NCC_all == -1) > 0:
        print('Number of weird NCC {}'.format(np.sum(NCC_all == -1)))
    if maxNCC == -1:
        print('VERY WEIRD ALL NCC ARE -1')
    idx = np.argmax(NCC_all)
    best_c1, best_c2 = np.ravel(searchx)[idx], np.ravel(searchy)[idx]
    return best_c1, best_c2, maxNCC
