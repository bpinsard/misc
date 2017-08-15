import numpy as np
import nibabel as nb

def wb_to_tss(lh_ctx_file, rh_ctx_file, sc_file):
    import numpy as np
    import nibabel as nb
    ds = np.hstack([np.asarray([da.data for da in nb.load(f).darrays]) for f in [lh_ctx_file, rh_ctx_file, sc_file]])
    return ds

def ddiff_var_moco(in_file):
    import numpy as np
    import h5py
    ts = h5py.File(in_file,'r')
    tss = np.asarray(ts['FMRI/DATA']).T
    out_fname = 'ddiff.npz'
    np.savez(out_fname,diffvar=np.nanstd(tss,0), ddiffvar=np.nanstd(np.diff(tss,1,0),0))
    return out_fname
    
def ddiff_var_wb(lh_ctx_file, rh_ctx_file, sc_file):
    import numpy as np
    from generic_pipelines.moco_eval import wb_to_tss
    tss = wb_to_tss(lh_ctx_file, rh_ctx_file, sc_file)
    out_fname = 'ddiff.npz'
    np.savez(out_fname,diffvar=np.nanstd(tss,0), ddiffvar=np.nanstd(np.diff(tss,1,0),0))
    return out_fname

def corr_delta_motion_moco(in_file, motion_file, nslabs=None):
    import numpy as np
    import h5py
    from generic_pipelines.moco_eval import corr_delta_motion
    ts = h5py.File(in_file,'r')
    tss = np.asarray(ts['FMRI/DATA']).T
    motion = np.loadtxt(motion_file).reshape(-1,nslabs,6).mean(1)
    corr = corr_delta_motion(tss, motion)
    out_fname = 'corr.npy'
    np.save(out_fname,corr)
    return out_fname

def corr_delta_motion_wb(lh_ctx_file, rh_ctx_file, sc_file, motion_file):
    import numpy as np
    from generic_pipelines.moco_eval import corr_delta_motion, wb_to_tss
    tss = wb_to_tss(lh_ctx_file, rh_ctx_file, sc_file)
    motion = np.loadtxt(motion_file)
    corr = corr_delta_motion(tss, motion)
    out_fname = 'corr.npy'
    np.save(out_fname,corr)
    return out_fname

def corr_delta_motion(tss, motion):
    dtss = np.diff(tss,1,0)
    drms = np.sqrt(np.square(np.diff(motion,1,0)).sum(-1))
    drms = np.abs(drms)
    dtss[:] = np.abs(dtss)
    cov = drms.dot(dtss)/len(drms)
    return cov


def reg_delta_motion_moco(in_file, motion_file, nslabs=None):
    import numpy as np
    import h5py
    from generic_pipelines.moco_eval import reg_delta_motion
    ts = h5py.File(in_file,'r')
    tss = np.asarray(ts['FMRI/DATA']).T
    motion = np.loadtxt(motion_file).reshape(-1,nslabs,6).mean(1)
    betas = reg_delta_motion(tss, motion)
    out_fname = 'betas.npy'
    np.save(out_fname,betas)
    return out_fname

def reg_delta_motion_wb(lh_ctx_file, rh_ctx_file, sc_file, motion_file):
    import numpy as np
    from generic_pipelines.moco_eval import reg_delta_motion, wb_to_tss
    tss = wb_to_tss(lh_ctx_file, rh_ctx_file, sc_file)
    motion = np.loadtxt(motion_file)
    betas = reg_delta_motion(tss, motion)
    out_fname = 'betas.npy'
    np.save(out_fname,betas)
    return out_fname

def reg_delta_motion(tss, motion):
    if len(tss) < 10:
        return 0
    dtss = np.diff(tss,1,0)
    dtss[:] = np.abs(dtss)
    dtss[~np.isfinite(dtss)]=0
    drms = np.sqrt(np.square(np.diff(motion,1,0)).sum(-1))
    drms /= drms.std()
    #regs = np.asarray([drms,np.ones(len(drms))]).T
    regs = drms[:,np.newaxis]
    betas = np.linalg.lstsq(regs, dtss)[0]
    return betas

def resting_dmn(sub, ses, in_file=None,
                lh_ctx_file=None, rh_ctx_file=None, sc_file=None,
                schedule_file=None):
    from pipe_hbn_ssi import wb_to_tss
    import os, sys
    import numpy as np
    sys.path.append('/home/bpinsard/data/projects/CoRe')
    import core.mvpa.dataset as cds
    from nipy.modalities.fmri.glm import GeneralLinearModel
    import scipy.ndimage    
    from mvpa2.datasets import Dataset

    sched = np.loadtxt(
        schedule_file, 
        converters = {0:int,1:int,2:str,3:int,4:str,5:str},
        dtype=np.object,
        skiprows=1)
    idx = sched[:,1].tolist().index(ses)
    #scan_no = sched[idx,2].split('-').index('Rest')
    if in_file is None:
        scan_no = [i for i,n in enumerate(lh_ctx_file) if 'RESTING' in n]
    else:
        scan_no = [i for i,n in enumerate(in_file) if 'RESTING' in n]
    scan_no = scan_no[0]
    
    if in_file is None:
        inf = lh_ctx_file[scan_no]
        print(inf)
        ds = Dataset(wb_to_tss(lh_ctx_file[scan_no], rh_ctx_file[scan_no], sc_file[scan_no]))
    else:
        inf = in_file[scan_no]
        print(inf)
        ds = cds.ds_from_ts(in_file[scan_no])
    #cds.preproc_ds(ds, detrend=True)
    ds.samples -= scipy.ndimage.gaussian_filter1d(ds.samples,sigma=8,axis=0,truncate=2)
    
    seed_roi = 9
    cds.add_aparc_ba_fa(ds, sub, pproc_tpl=os.path.join(pipe_hbn_ssi.proc_dir,'moco_multiband','surface_32k','_sub_%d'))
    roi_mask = np.logical_or(ds.fa.aparc==seed_roi+11100, ds.fa.aparc==seed_roi+12100)
    
    mean_roi_ts = ds.samples[:,roi_mask].mean(1)
    mean_roi_ts -= mean_roi_ts.mean()
    
    mtx = np.asarray([mean_roi_ts, np.ones(ds.nsamples)]).T
        
    glm = GeneralLinearModel(mtx)
    glm.fit(ds.samples,model='ols')
    contrast = glm.contrast([1,0], contrast_type='t')
    
    out_file = os.path.abspath('sub%d_ses%d_connectivity_results.npz'%(sub,ses))
    np.savez_compressed(out_file, contrast=contrast, mean_roi_ts=mean_roi_ts)
    #return contrast
    return out_file, inf
