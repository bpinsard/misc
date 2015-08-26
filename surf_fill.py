import numpy as np
import nibabel as nb
import nibabel.gifti
import scipy.ndimage

def surf_fill_vtk(vertices, polys, mat, shape):

    import vtk
    from vtk.util import numpy_support


    voxverts = nb.affines.apply_affine(np.linalg.inv(mat), vertices)
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(len(voxverts))
    for i,pt in enumerate(voxverts):
        points.InsertPoint(i, pt)

    tris  = vtk.vtkCellArray()
    for vert in polys:
        tris.InsertNextCell(len(vert))
        for v in vert:
            tris.InsertCellPoint(v)

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.SetPolys(tris)
    del points, tris

    whiteimg = vtk.vtkImageData()
    whiteimg.SetDimensions(shape)
    if vtk.VTK_MAJOR_VERSION <= 5:
        whiteimg.SetScalarType(vtk.VTK_UNSIGNED_CHAR)
    else:
        info = vtk.vtkInformation()
        whiteimg.SetPointDataActiveScalarInfo(info, vtk.VTK_UNSIGNED_CHAR, 1)

    ones = np.ones(np.prod(shape),dtype=np.uint8)
    whiteimg.GetPointData().SetScalars(numpy_support.numpy_to_vtk(ones))
    
    pdtis = vtk.vtkPolyDataToImageStencil()
    if vtk.VTK_MAJOR_VERSION <= 5:
        pdtis.SetInput(pd)
    else:
        pdtis.SetInputData(pd)

    pdtis.SetOutputWholeExtent(whiteimg.GetExtent())
    pdtis.Update()

    imgstenc = vtk.vtkImageStencil()
    if vtk.VTK_MAJOR_VERSION <= 5:
        imgstenc.SetInput(whiteimg)
        imgstenc.SetStencil(pdtis.GetOutput())
    else:
        imgstenc.SetInputData(whiteimg)
        imgstenc.SetStencilConnection(pdtis.GetOutputPort())
    imgstenc.SetBackgroundValue(0)

    imgstenc.Update()
    
    data = numpy_support.vtk_to_numpy(
        imgstenc.GetOutput().GetPointData().GetScalars()).reshape(shape).transpose(2,1,0)
    del pd,voxverts,whiteimg,pdtis,imgstenc
    return data

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
        fill = surf_fill_vtk(vertices, tris,
                         parc.get_affine(), parc.shape, pve_voxsize)
        pve = reduce(
            lambda x,y: x+fill[0][y[0]::subdiv,y[1]::subdiv,y[2]::subdiv],
            np.mgrid[:subdiv,:subdiv,:subdiv].reshape(3,-1).T,0
            ).astype(np.float32)
        pve /= float(subdiv**3)
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
        sigma=voxsize)
    subcort_smooth = scipy.ndimage.gaussian_filter(
        group_rois([10,11,12,13,26,49,50,51,52,58]).astype(np.float32),
        sigma=voxsize)
    wm_smooth = scipy.ndimage.gaussian_filter(
        group_rois([7,16,28,46,60,85,192,
                    250,251,252,253,254,255]).astype(np.float32),
        sigma=voxsize)

    # remove csf at the end of brainstem for streamlines to medulla
    # suppose hcp orientation storage
    bs = (parc_data==16).any(0).any(0)
    lbs = np.where(bs)[0][0]+3
    outer_csf = np.logical_and(mask_data>0, parc_data==0)
    outer_csf[...,:lbs] = 0
    csf_smooth = scipy.ndimage.gaussian_filter(
        np.logical_or(
            group_rois([4,5,14,15,24,30,31,43,44,62,63,72]),
            outer_csf).astype(np.float32),
        sigma=voxsize)
    csf_smooth[...,:lbs] = 0

    wm =  wm_pve+wm_smooth-csf_smooth-subcort_smooth
    wm[wm>1] = 1
    wm[wm<0] = 0
    
    gm = gm_pve-wm_pve-wm-subcort_smooth+gm_smooth
    gm[gm<0] = 0
    
    tt5 = np.concatenate([gm[...,np.newaxis],
                          subcort_smooth[...,np.newaxis],
                          wm[...,np.newaxis],
                          csf_smooth[...,np.newaxis],
                          np.zeros(parc.shape+(1,),dtype=np.float32)],3)

    tt5[...,:lbs,:] = 0

    tt5/=tt5.sum(-1)[...,np.newaxis]
    tt5[np.isnan(tt5)]=0

    return nb.Nifti1Image(tt5.astype(np.float32),parc.get_affine())


def read_surf(fname, surf_ref):
    if fname[-4:] == '.gii':
        gii = nb.gifti.read(fname)
        return gii.darrays[0].data, gii.darrays[1].data
    else:
        verts,tris =  nb.freesurfer.read_geometry(fname)
        ras2vox = np.array([[-1,0,0,128],[0,0,-1,128],[0,1,0,128],[0,0,0,1]])
        surf2world = surf_ref.get_affine().dot(ras2vox)
        verts[:] = nb.affines.apply_affine(surf2world, verts)
        return verts, tris

def make_5tt(parc_file, lh_white, rh_white, lh_pial, rh_pial, subdiv=4):
    parc = nb.load(parc_file)
    voxsize = np.asarray(parc.header.get_zooms()[:3])
    parc_data = parc.get_data()
    lh_wm = read_surf(lh_white, parc)
    rh_wm = read_surf(rh_white, parc)
    lh_gm = read_surf(lh_pial, parc)
    rh_gm = read_surf(rh_pial, parc)

    
    def fill_hemis(lh_surf,rh_surf):
        vertices = np.vstack([lh_surf[0],rh_surf[0]])
        tris = np.vstack([lh_surf[1],
                          rh_surf[1]+lh_surf[0].shape[0]])
        pve_voxsize = voxsize/float(subdiv)
        mat = parc.affine.dot(np.diag([1/float(subdiv)]*3+[1]))
        shape = np.asarray(parc.shape)*subdiv
        fill = surf_fill_vtk(vertices, tris, mat, shape)
        pve = reduce(
            lambda x,y: x+fill[y[0]::subdiv,y[1]::subdiv,y[2]::subdiv],
            np.mgrid[:subdiv,:subdiv,:subdiv].reshape(3,-1).T,0
            ).astype(np.float32)
        pve /= float(subdiv**3)
        return pve
    wm_pve = fill_hemis(lh_wm,rh_wm)
    gm_pve = fill_hemis(lh_gm,rh_gm)

    
    def group_rois(rois_ids):
        m = np.zeros(parc.shape, dtype=np.bool)
        for i in rois_ids:
            np.logical_or(parc_data==i, m, m)
        return m
        
    gm_rois = group_rois([8,47,17,18,53,54]).astype(np.float32)
    gm_smooth = scipy.ndimage.gaussian_filter(gm_rois, sigma=voxsize)

    subcort_rois = group_rois([10,11,12,13,26,49,50,51,52,58]).astype(np.float32)
    subcort_smooth = scipy.ndimage.gaussian_filter(subcort_rois, sigma=voxsize)
    
    wm_rois = group_rois([7,16,28,46,60,85,192,88,
                         250,251,252,253,254,255]).astype(np.float32)

    wm_smooth = scipy.ndimage.gaussian_filter(wm_rois,sigma=voxsize)

    # remove csf at the end of brainstem for streamlines to medulla
    # suppose hcp orientation storage
    bs_mask = parc_data==16
    bs_vdc_dil = scipy.ndimage.morphology.binary_dilation(group_rois([16,60,28]), iterations=2)
    # mask of boundary between brainstem and cerebellar gray matter
    bs_vdc_excl = np.logical_and(bs_vdc_dil, np.logical_not(group_rois([16,7,46,60,28,10,49,2,41,0])))

    lbs = np.where((bs_mask).any(-1).any(0))[0][-1]-3

    parc_data_mask = parc_data>0
    outer_csf = np.logical_and(
        np.logical_not(parc_data_mask),
        scipy.ndimage.morphology.binary_dilation(parc_data_mask))

    ## create a fake GM rois at the end of brainstem for cerebro-spinal tracking
    nb.save(nb.Nifti1Image(outer_csf.astype(np.int32),parc.affine),'outer_csf.nii')

    csf_rois = group_rois([4,5,14,15,24,30,31,43,44,62,63,72])
    nb.save(nb.Nifti1Image(csf_rois.astype(np.int32),parc.affine),'csf_rois.nii')

    csf_smooth = scipy.ndimage.gaussian_filter(
        np.logical_or(csf_rois, outer_csf).astype(np.float32),
        sigma=voxsize)
    nb.save(nb.Nifti1Image(csf_smooth,parc.affine),'csf_smooth.nii')


    bs_roi = csf_smooth.copy()
    bs_roi[...,:lbs,:] = 0 
    csf_smooth[...,lbs:,:] = 0
    wm_smooth[...,lbs:,:] = 0

    # add csf around brainstem and ventral DC to remove direct connection to gray matter
    csf_smooth[bs_vdc_excl] += gm_smooth[bs_vdc_excl]
    gm_smooth[bs_vdc_excl] = 0

    mask88 = parc_data==88    
    print csf_smooth[mask88].sum(), subcort_smooth[mask88].sum()

#    csf_smooth -= wm_smooth
#    csf_smooth[csf_smooth<0]=0

    nb.save(nb.Nifti1Image(wm_pve,parc.affine),'wm_pve.nii')
    nb.save(nb.Nifti1Image(wm_smooth,parc.affine),'wm_smooth.nii')
    nb.save(nb.Nifti1Image(subcort_smooth,parc.affine),'subcort_smooth.nii')

    wm = wm_pve+wm_smooth-csf_smooth-subcort_smooth
    wm[wm>1] = 1
    wm[wm<0] = 0

    print 267, np.count_nonzero(wm[mask88])
    
    gm = gm_pve-wm_pve-wm-subcort_smooth+gm_smooth+bs_roi
    gm[gm<0] = 0
 
    
   
    tt5 = np.concatenate([gm[...,np.newaxis],
                          subcort_smooth[...,np.newaxis],
                          wm[...,np.newaxis],
                          csf_smooth[...,np.newaxis],
                          np.zeros(parc.shape+(1,),dtype=np.float32)],3)


    tt5/=tt5.sum(-1)[...,np.newaxis]
    tt5[np.isnan(tt5)]=0

    return nb.Nifti1Image(tt5.astype(np.float32),parc.affine)
