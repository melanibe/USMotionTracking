import numpy as np

# Find pixels around a given center


def find_search_pixel(c1, c2, width=100):
    xax = np.linspace(c1-width/2, c1+width/2, 51, dtype='int')
    yax = np.linspace(c2-width/2, c2+width/2, 51, dtype='int')
    #xax = np.arange(int(c1)-25, int(c1)+26, step=1)
    #yax = np.arange(int(c2)-25, int(c2)+26, step=1)
    # all the x,y in the template centered around c1, c2
    return np.meshgrid(xax, yax)


def find_template_pixel(c1, c2, width=60):
    w = width//2
    xax = np.arange(int(c1-1)-w, int(c1-1)+w+1, step=1)
    yax = np.arange(int(c2-1)-w, int(c2-1)+w+1, step=1)
    # all the x,y in the template centered around c1, c2
    return np.meshgrid(xax, yax)


def find_new_template_center_NCC(c1, c2, im1, im2, width=60, c1_init=None, c2_init=None):
    searchx, searchy = find_search_pixel(c1, c2)
    if c1_init is None:
        xv, yv = find_template_pixel(c1, c2, width)
    else:
        xv, yv = find_template_pixel(c1_init, c2_init, width)
    NCC_all = []
    for i, j in zip(np.ravel(searchx), np.ravel(searchy)):
        # print(i,j)
        tmp_x, tmp_y = find_template_pixel(i, j, width)
        #fig, ax = plt.subplots(1)
        # ax.imshow(im2)
        #ax.scatter(tmp_x, tmp_y)
        # plt.show()
        x1 = np.ravel(im1[np.ravel(yv), np.ravel(xv)])
        x2 = np.ravel(im2[np.ravel(tmp_y), np.ravel(tmp_x)])
        x1 = x1 - np.mean(x1)
        x2 = x2 - np.mean(x2)
        num = np.sum(x1*x2)
        denom = np.sqrt(np.sum(x1**2)*np.sum(x2**2))
        NCC_all.append(num/denom)
    maxNCC = np.max(NCC_all)
    idx = np.argmax(NCC_all)
    best_c1, best_c2 = np.ravel(searchx)[idx], np.ravel(searchy)[idx]
    return best_c1, best_c2, maxNCC


def global_template_search(c1,
                           c2,
                           c1_init,
                           c2_init,
                           im_prev,
                           im_current,
                           im_init,
                           width=60):
    best_c1_1, best_c2_1, maxNCC_1 = find_new_template_center_NCC(c1, c2,
                                                                  im_prev,
                                                                  im_current,
                                                                  width)
    return best_c1_1, best_c2_1, maxNCC_1