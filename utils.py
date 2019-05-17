import logging
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
'''
Mélanie Bernhardt - ETH Zurich
CLUST Challenge
'''

def get_logger(checkpoint_dir):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger('my_log')
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    # create formatter
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    log_filename = checkpoint_dir + '/logfile' + '.log'
    file_handler = logging.FileHandler(log_filename)
    # file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    return logger

def get_default_params(params_dict):
    if params_dict.get('width') is None:
        params_dict['width'] = 60
    if params_dict.get('n_epochs') is None:
        params_dict['n_epochs'] = 15
    if params_dict.get('h1') is None:
        params_dict['h1'] = 32
    if params_dict.get('h2') is None:
        params_dict['h2'] = 64
    if params_dict.get('h3') is None:
        params_dict['h3'] = 0
    if params_dict.get('embed_size') is None:
        params_dict['embed_size'] = 64
    if params_dict.get('dropout_rate') is None:
        params_dict['dropout_rate'] = 0
    if params_dict.get('use_batchnorm') is None:
        params_dict['use_batchnorm'] = True
    return params_dict

def plot_img_template(c1,c2,img, width=50, height=50):
    x = c1 - width/2
    y = c2 - height/2
    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.scatter(c1-1, c2-1, s=10)
    rect=Rectangle((x,y), width, height, fill=False)
    ax.add_patch(rect)
    plt.show()