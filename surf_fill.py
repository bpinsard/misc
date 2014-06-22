import numpy as np
from tvtk.api import tvtk


def surf_fill(vertices, polys, mat, shape, voxel_size=None):
    
    orig_voxel_size = np.sqrt((mat[:3,:3]**2).sum(0))
    if voxel_size is None:
        mat_out = mat
        voxel_size = orig_voxel_size
        origin = mat_out[:3,3]
    else:
        ratio = np.asarray(voxel_size)/orig_voxel_size
        mat_out = mat.copy()
        mat_out[:3,:3] = np.diag(ratio).dot(mat[:3,:3])
        shapef = shape/ratio
        shape = tuple(np.ceil(shapef).astype(np.int))
        delta = (shape-shapef)/2.
        origin = mat_out[:3,3]-mat_out[:3,:3].dot(np.diag(1./np.asarray(voxel_size))).dot(delta)
        mat_out[:3,3] = origin
    voxel_size2 = mat_out[:3,:3].dot(np.ones(3))

    pd = tvtk.PolyData(points=vertices, polys=polys)

    whiteimg = tvtk.ImageData()
    whiteimg.spacing = voxel_size2
    whiteimg.dimensions = shape
    whiteimg.extent = (0,shape[0]-1, 0,shape[1]-1, 0,shape[2]-1)
    whiteimg.origin = origin
    whiteimg.scalar_type = 'unsigned_char'
    whiteimg.point_data.scalars = np.ones(np.prod(shape), dtype=np.uint8)

    pdtis = tvtk.PolyDataToImageStencil()
    pdtis.input = pd
    pdtis.output_origin = origin
    pdtis.output_spacing = voxel_size2
    pdtis.output_whole_extent = whiteimg.extent
    pdtis.update()

    imgstenc = tvtk.ImageStencil()
    imgstenc.input = whiteimg
    imgstenc.stencil = pdtis.output
    imgstenc.background_value = 0
    imgstenc.update()
    
    data = imgstenc.output.point_data.scalars.to_array()
    return data.reshape(shape[::-1]).transpose(2,1,0), mat_out

import nibabel as nb
import nibabel.gifti
import scipy.ndimage

def hcp_5tt(parc_file, mask_file,
            lh_white, rh_white, lh_pial, rh_pial, subdiv=4):
    parc = nb.load(parc_file)
    mask = nb.load(mask_file)
    mask_data = mask.get_data()
    voxsize = np.asarray(parc.header.get_zooms()[:3])
    parc_data = parc.get_data()
    lh_wm = nb.gifti.read(lh_white)
    rh_wm = nb.gifti.read(rh_white)
    lh_gm = nb.gifti.read(lh_pial)
    rh_gm = nb.gifti.read(rh_pial)

    
    def fill_hemis(lh_surf,rh_surf):
        vertices = np.vstack([lh_surf.darrays[0].data,rh_surf.darrays[0].data])
        tris = np.vstack([lh_surf.darrays[1].data,
                          rh_surf.darrays[1].data+lh_surf.darrays[0].dims[0]])
        pve_voxsize = voxsize/float(subdiv)
        fill = surf_fill(vertices, tris,
                         parc.get_affine(), parc.shape, pve_voxsize)
        pve = reduce(
            lambda x,y: x+fill[0][y[0]::subdiv,y[1]::subdiv,y[2]::subdiv],
            np.mgrid[:subdiv,:subdiv,:subdiv].reshape(3,-1).T,0
            ).astype(np.float32)/float(subdiv**3)
        return pve
    wm_pve = fill_hemis(lh_wm,rh_wm)
    gm_pve = fill_hemis(lh_gm,rh_gm)
    
    def group_rois(rois_ids):
        m = np.zeros(parc.shape, dtype=np.bool)
        for i in rois_ids:
            np.logical_or(parc_data==i, m, m)
        return m
        

    gm_smooth = scipy.ndimage.gaussian_filter(
        group_rois([8,47,17,18,53,54]).astype(np.float32),
        sigma=voxsize/2.)
    subcort_smooth = scipy.ndimage.gaussian_filter(
        group_rois([10,11,12,13,26,49,50,51,52,58]).astype(np.float32),
        sigma=voxsize/2.)
    wm_smooth = scipy.ndimage.gaussian_filter(
        group_rois([7,16,28,46,60,85,192,
                    250,251,252,253,254,255]).astype(np.float32),
        sigma=voxsize/2.)
    csf_smooth = scipy.ndimage.gaussian_filter(
        np.logical_or(
            group_rois([4,5,14,15,24,30,31,43,44,62,63,72]),
            np.logical_and(mask_data>0, parc_data==0)).astype(np.float32),
        sigma=voxsize)

#    bs = parc_data==16
#    np.where(bs.any(0).any(1))

    wm =  wm_pve+wm_smooth-csf_smooth-subcort_smooth
    wm[wm>1] = 1
    wm[wm<0] = 0
    
    gm = gm_pve-wm_pve-wm-subcort_smooth+gm_smooth
    gm[gm<0] = 0
    
    tt5 = np.concatenate([gm[...,np.newaxis],
                          subcort_smooth[...,np.newaxis],
                          wm[...,np.newaxis],
                          csf_smooth[...,np.newaxis],
                          np.zeros(parc.shape+(1,))],3)

    tt5/=tt5.sum(-1)[...,np.newaxis]
    tt5[np.isnan(tt5)]=0

    return nb.Nifti1Image(tt5,parc.get_affine())
    
    
    
    

