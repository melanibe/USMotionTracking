import logging

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