# loadmat(): 
# thanks to http://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

import scipy.io as spio
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle

def changeExt(filename_in, extension_old, extension_new):
    ext1 = extension_old[::-1]
    ext2 = extension_new
    if not ext1.endswith('.'):
        ext1 += '.'
    if not ext2.startswith('.'):
        ext2 = '.' + ext2
    fname = filename_in[::-1]
    if fname.startswith(ext1):
        fname = fname.replace(ext1, '', 1)
    fname = fname[::-1] + ext2
    return fname

def mat2txt(filename_in, filename_out=None):
    assert filename_in.endswith('.mat'), '*.mat file must be provided'
    if filename_out is None:
        filename_out = changeExt(filename_in, '.mat', '.txt')
    else:
        if not filename_out.endswith('.txt'):
            filename_out += '.txt'
    df = loadmat(filename_in)
    with open(filename_out, 'wb') as f:
        pickle.dump(df, f)
    return
    
def txt2mat(filename_in, filename_out=None):
    assert filename_in.endswith('.txt'), '*.txt file must be provided'
    if filename_out is None:
        filename_out = changeExt(filename_in, '.txt', '.mat')
    else:
        if not filename_out.endswith('.mat'):
            filename_out += '.mat'
    with open(filename_in, 'rb') as f:
        df = pickle.loads(f.read())
    savemat(df, filename_in)
    return

def savemat(df, filename):
    assert isinstance(df, dict), 'Data must be a python dictionary istance'
    assert isinstance(filename, str), 'Filename must be a string instance'
    addExt = not filename.endswith('.mat')
    spio.savemat(filename, df, appendmat=addExt, format='5',
        long_field_names=False, do_compression=True, oned_as='row'
    )
    return 

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict        

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        elif isinstance(elem,np.ndarray):
            dict[strg] = _tolist(elem)
        else:
            dict[strg] = elem
    return dict

def _tolist(ndarray):
    '''
    A recursive function which constructs lists from cellarrays 
    (which are loaded as numpy ndarrays), recursing into the elements
    if they contain matobjects.
    '''
    elem_list = []            
    for sub_elem in ndarray:
        if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
            elem_list.append(_todict(sub_elem))
        elif isinstance(sub_elem,np.ndarray):
            elem_list.append(_tolist(sub_elem))
        else:
            elem_list.append(sub_elem)
    return elem_list
