from cPickle import dumps
import operator

# wrap builtin methods
# when used for connection function, supplementary args needs to be in a list
def wrap(method):
    return 'def %s_wrapped(*args,**kwargs): from %s import %s ;return %s(*args,**kwargs)'%(method.__name__,method.__module__,method.__name__,method.__name__)
#    return dumps('def %s_wrapped(*args,**kwargs): from %s import %s ;return %s(*args,**kwargs)'%(method.__name__,method.__module__,method.__name__,method.__name__))

# recursive item getter first args are evaluated on top of the structure
def getitem_rec(*args):
    import operator
    if len(args)>2:
        if isinstance(args[1],slice):
            return [getitem_rec(a,*args[2:]) for a in \
                        operator.getitem(args[0],args[1])]
        else:
            return getitem_rec(operator.getitem(args[0],args[1]),*args[2:])
    else:
        return operator.getitem(args[0],args[1])

#fname_presuffix removing path to create filename in connect function
def fname_presuffix_basename(*args,**kwargs):
    import os
    import nipype.utils.filemanip as fmanip
    if isinstance(args[0],list):
        return [os.path.basename(f) for f in fmanip.fnames_presuffix(*args,**kwargs)]
    return os.path.basename(fmanip.fname_presuffix(*args,**kwargs))

def wrap_iterate(method):
    return dumps('def %s_wrapped(*args,**kwargs): from %s import %s ; return [ %s((arg0,)+args[1:],**kwargs) for arg0 in args[0]]'%(method.__name__,method.__module__,method.__name__,method.__name__))
    
class NoScan(Exception):
    def __init__(self, scan):
        self.scan = scan
    def __str__(self):
        return 'Warning : no scan %d'%self.scan

def scan_switch(in_files,scan):
    from generic_pipelines.utils import NoScan
    if isinstance(in_files,str) and scan<1:
        return in_files
    elif isinstance(in_files,list) and scan<len(in_files):
        return in_files[scan]
    else:
        raise NoScan(scan)


        
def wildcard(p,s=slice(0,None),wc='*'):
    import os
    if isinstance(p,list):
        if isinstance(s,list):
            return [os.path.join(p[i],wc) for i in s]
        elif isinstance(s,slice):
            return [os.path.join(pp,wc) for pp in p[s]]
        elif isinstance(s,int):
            return os.path.join(p[s],wc)
    return os.path.join(p,wc)



def xfm2mat(xfm):
    import os
    import numpy as np
    mat=np.loadtxt(xfm, skiprows=5,usecols=range(4))
    np.savetxt("out.mat",np.vstack([mat,[0,0,0,1]]))
    return os.path.abspath("out.mat")
