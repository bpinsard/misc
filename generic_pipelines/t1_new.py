import os
from nipype.interfaces import spm, fsl, afni, utility, nipy, utility, freesurfer

import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.workflows.rsfmri.spm as spmrest
import nipype.interfaces.nipy.utils as nipyutils
import nipype.interfaces.nipy.preprocess as nipypp
import nipype.interfaces.dcmstack as np_dcmstack
import nipype.utils.filemanip as fmanip
import nipype.pipeline.file_proxy as fileproxy
from .utils import *

def dicom_convert_ppl(name='t1_dicom_cvt', crop_t1=True):
    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['dicom_files','out_file_format',
                    'voxel_order','meta_force_add']),
        run_without_submitting = True,
        name='inputspec')

    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['t1_nifti']),
        run_without_submitting = True,
        name='outputspec')

    n_convert_t1_dicom = pe.Node(
        np_dcmstack.DCMStackAnatomical(),
        name='convert_t1_dicom')

    n_autobox_t1 = pe.Node(
        afni.Autobox(),
        name='autobox_t1')

    n_crop_t1 = pe.Node(
        nipy.Crop(out_file='%s_crop.nii',outputtype='NIFTI'),
        name = 'crop_t1')

    w = pe.Workflow(name=name)
    
    def zmax2keep(z):
        return max(0,z-174)
    wmax=wrap(max)

    w.connect([
       (inputnode, n_convert_t1_dicom,[('dicom_files','dicom_files'),
                                       ('out_file_format','out_file_format'),
                                       ('voxel_order','voxel_order'),
                                       ('meta_force_add','meta_force_add')]),
       ])
    if crop_t1:
        w.connect([
                (n_convert_t1_dicom, n_autobox_t1, [('nifti_file','in_file')]),
                (n_convert_t1_dicom, n_crop_t1, [('nifti_file','in_file')]),
                (n_autobox_t1, n_crop_t1, [
                        (('x_min',wmax,[0]),'x_min'),('x_max','x_max'),
                        (('y_min',wmax,[0]),'y_min'),('y_max','y_max'),
                        ('z_max','z_max'),(('z_max',zmax2keep),'z_min')]),
                (n_autobox_t1, outputnode, [('out_file','t1_nifti')]),
                ])
    else:
        w.connect(n_convert_t1_dicom,'nifti_file',outputnode,'t1_nifti'),

    return w


def t1_pipeline(name='t1_preproc'):
    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['t1_file']),
        run_without_submitting = True,
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['mask','corrected_t1','corrected_t1_brain']),
        run_without_submitting = True,
        name='outputspec')

    spm_path = spm.Info.version()['path']

    tpm = os.path.join(spm_path,'toolbox','Seg','TPM.nii')
    n_newsegment_t1 = pe.Node(
        spm.NewSegment(
            write_deformation_fields=[False,True],
            channel_info = (0.0001,60,(True,True)),
            tissues=[
                ((tpm,1),2,(True,True),(True,True)),
                ((tpm,2),2,(True,True),(True,True)),
                ((tpm,3),2,(True,False),(True,True)),
                ((tpm,4),3,(True,False),(False,False)),
                ((tpm,5),4,(True,False),(False,False)),
                ((tpm,6),2,(False,False),(False,False)),
                ]),
        name='newsegment_t1')

    n_seg2mask = pe.Node(
        fsl.MultiImageMaths(
            output_type='NIFTI', output_datatype='char',
            op_string=' -add %s -add %s -thr 0.8 -bin -eroF -dilF -dilF'),
        name='seg2mask')

    n_autobox_mask = pe.Node(
        afni.Autobox(padding=3),
        name='autobox_mask')

    n_merge_crop = pe.Node(
        utility.Merge(3),
        name='merge_crop')

    n_crop_all = pe.MapNode(
        nipy.Crop(out_file='%s_crop.nii.gz', outputtype='NIFTI_GZ'),
        iterfield = ['in_file'],
        name='crop_all')

    w = pe.Workflow(name=name)
    
    w.connect([
        (inputnode, n_newsegment_t1, [('t1_file','channel_files')]),

        (n_newsegment_t1, n_seg2mask, [
            (('native_class_images',getitem_rec,0,0), 'in_file'),
            (('native_class_images',getitem_rec,slice(1,3),0),'operand_files')]),
        (inputnode,n_seg2mask,[
            (('t1_file',fname_presuffix_basename,'','_mask','.'),'out_file')]),

        (n_seg2mask, n_autobox_mask, [('out_file','in_file')]),
        (n_newsegment_t1, n_merge_crop, [
                    ('bias_corrected_images','in1'),
                    (('native_class_images',utility.select,[0,1,2]),'in2')]),
        (n_seg2mask, n_merge_crop, [('out_file','in3')]),
        (n_merge_crop, n_crop_all, [(('out',utility.flatten),'in_file')]),
        (n_autobox_mask, n_crop_all, [('x_min','x_min'),('x_max','x_max'),
                                      ('y_min','y_min'),('y_max','y_max'),
                                      ('z_min','z_min'),('z_max','z_max'),]),
        (n_seg2mask, outputnode,[('out_file','mask')]),
        (n_newsegment_t1,outputnode,[('bias_corrected_images','corrected_t1')])
        ])
    return w


def seg2mask(tissues_files,out_file=None):
    import nibabel as nb, numpy as np, os
    import scipy.ndimage
    from generic_pipelines.utils import fname_presuffix_basename
    niis = [nb.load(f) for f in tissues_files]
    data = [n.get_data() for n in niis]
    mask = np.logical_or(data[0]>.5,data[1]>.5)
    np.logical_or(data[2]>.9,mask,mask)
    np.logical_or(np.logical_and(data[0]+data[1]>.2,data[2]>.5),mask,mask)
    scipy.ndimage.binary_fill_holes(mask,None,mask)
    if out_file==None:
        out_file=fname_presuffix_basename(tissues_files[0],suffix='_mask')
    out_file = os.path.abspath(os.path.join(os.getcwd(),out_file))
    nb.save(nb.Nifti1Image(mask.astype(np.uint8),niis[0].get_affine()),
            out_file)
    del niis, data, mask
    return out_file
    
def t1_vbm_pipeline(name='t1_preproc'):
    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['t1_file']),
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['mask','corrected_t1','corrected_t1_brain']),
        name='outputspec')

    spm_path = spm.Info.version()['path']

    tpm = os.path.join(spm_path,'toolbox','Seg','TPM.nii')
    dartel_tpl = os.path.join(spm_path,'toolbox','vbm8','Template_1_IXI550_MNI152.nii')
    n_vbm_segment = pe.Node(
        spm.VBMSegment(
            tissues=tpm,dartel_template=dartel_tpl,
            bias_corrected_native = True,
            gm_native = True, gm_dartel = 1,
            wm_native = True, wm_dartel = 1,
            csf_native = True,
            pve_label_native = True,
            deformation_field = (True, False)),
        name='vbm_segment')

    n_seg2mask = pe.Node(
        fsl.MultiImageMaths(
            output_type='NIFTI', output_datatype='char',
            op_string=' -add %s -add %s -thr 0.8 -bin -eroF -dilF -dilF'),
        name='seg2mask')

    n_seg2mask = pe.Node(
        utility.Function(input_names=['tissues_files','out_file'],
                      output_names=['out_file'],
                      function=seg2mask),
        name='seg2mask')

    n_autobox_mask = pe.Node(
        afni.Autobox(padding=3),
        name='autobox_mask')

    n_merge_crop = pe.Node(
        utility.Merge(3),
        name='merge_crop')

    n_crop_all = pe.MapNode(
        nipy.Crop(out_file='%s_crop.nii.gz',
                  outputtype='NIFTI_GZ'),
        iterfield = ['in_file'],
        name='crop_all')

    w = pe.Workflow(name=name)
    
    w.connect([
        (inputnode, n_vbm_segment, [('t1_file','in_files')]),

        (n_vbm_segment, n_seg2mask, [
            (('native_class_images',utility.flatten), 'tissues_files')]),
#        (n_vbm_segment, n_seg2mask, [
#            (('native_class_images',getitem_rec,0,0), 'in_file'),
#            (('native_class_images',getitem_rec,slice(1,3),0),'operand_files')]),
        (inputnode,n_seg2mask,[
            (('t1_file',fname_presuffix_basename,'','_mask','.'),'out_file')]),

        (n_seg2mask, n_autobox_mask, [('out_file','in_file')]),
        (n_vbm_segment, n_merge_crop, [
                    ('bias_corrected_images','in1'),
                    (('native_class_images',utility.select,[0,1,2]),'in2')]),
        (n_seg2mask, n_merge_crop, [('out_file','in3')]),
        (n_merge_crop, n_crop_all, [(('out',utility.flatten),'in_file')]),
        (n_autobox_mask, n_crop_all, [('x_min','x_min'),('x_max','x_max'),
                                      ('y_min','y_min'),('y_max','y_max'),
                                      ('z_min','z_min'),('z_max','z_max'),]),
        (n_seg2mask, outputnode,[('out_file','mask')]),
        (n_vbm_segment,outputnode,[('bias_corrected_images','corrected_t1')])
        ])
    return w

def fs_seg2mask(parc_file,out_file=None):
    import nibabel as nb, numpy as np, os
    import scipy.ndimage
    from generic_pipelines.utils import fname_presuffix_basename
    nii = nb.load(parc_file)
    op = ((np.mgrid[:5,:5,:5]-2.0)**2).sum(0)<=4
    mask = scipy.ndimage.binary_closing(nii.get_data()>0,op,iterations=2)
    scipy.ndimage.binary_fill_holes(mask,output=mask)
    if out_file==None:
        out_file=fname_presuffix_basename(parc_file,suffix='_mask')
    out_file = os.path.abspath(os.path.join(os.getcwd(),out_file))
    nb.save(nb.Nifti1Image(mask.astype(np.uint8),nii.get_affine()),out_file)
    del nii, mask, op
    return out_file

def t1_freesurfer_pipeline(name='t1_preproc'):
    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['t1_file','subject_id']),
        run_without_submitting = True,
        name='inputspec')

    n_freesurfer = pe.Node(
        interface=freesurfer.ReconAll(directive='all',args='-use-gpu'),
        name='freesurfer',)

    n_fs_seg2mask = pe.Node(
        utility.Function(input_names=['parc_file','out_file'],
                         output_names=['out_file'],
                         function=fs_seg2mask),
        name='fs_seg2mask')

    n_autobox_mask = pe.Node(
        afni.Autobox(padding=3,out_file='%s_crop.nii'),
        name='autobox_mask_fs')

    w = pe.Workflow(name=name)
    w.connect([
            (inputnode, n_freesurfer, [
                    ('t1_file','T1_files'),
                    ('subject_id','subject_id')]),
            (n_freesurfer, n_fs_seg2mask,[
                    (('aparc_aseg',utility.select,1),'parc_file')]),
            (inputnode,n_fs_seg2mask,[
            (('t1_file',fname_presuffix_basename,'','_mask.nii','.',False),
             'out_file')]),
            (n_fs_seg2mask, n_autobox_mask,[('out_file','in_file')])
            ])
    return w

def extract_wm_regions(seg_file,rois_ids):
    import os
    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix
    seg = nb.load(seg_file)
    wm = np.zeros(seg.shape,np.uint8)
    for i in rois_ids:
        np.logical_or(wm,seg.get_data()==i,wm)
    out_fname=fname_presuffix(seg_file, suffix='_wm.nii.gz',
                              newpath=os.getcwd(), use_ext=False)
    nii_out = nb.Nifti1Image(wm,seg.get_affine())
    nb.save(nii_out, out_fname)
    del wm, seg, nii_out
    return out_fname


def extract_wm_surface(name='extract_wm_surface'):
    
    inputnode = pe.Node(
        utility.IdentityInterface(fields=['aseg']),
        run_without_submitting = True,
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(fields=['surface']),
        run_without_submitting = True,
        name='outputspec')

    n_extract_wm_regions = pe.Node(
        utility.Function(
            input_names=['seg_file','rois_ids'],
            output_names=['wm_file'],
            function=extract_wm_regions),
        name='extract_wm_regions')
    n_extract_wm_regions.inputs.rois_ids=[2,7,16,41,46, 251,252,253,254,255]

    n_tesselate = pe.Node(
        freesurfer.MRITessellate(label_value=1),
        name='tesselate')
    n_wm_main_component = pe.Node(
        freesurfer.ExtractMainComponent(out_file='rlh.%s.all'),
        name='wm_main_component')

    n_smooth_tessellation = pe.Node(
        freesurfer.SmoothTessellation(curvature_averaging_iterations=5,
                                      smoothing_iterations=5),
        name='smooth_tesselation')

    n_surf_decimate = pe.Node(
        freesurfer.Decimate(decimation_level=.2),
        name='surf_decimate')

    w = pe.Workflow(name=name)
    w.connect([
        (inputnode, n_extract_wm_regions, [('aseg','seg_file'),]),
        (n_extract_wm_regions, n_tesselate, [('wm_file','in_file'),]),
        (n_tesselate, n_wm_main_component, [('surface','in_file'),]),
        (n_wm_main_component, n_smooth_tessellation,[('out_file','in_file')]),
        (n_smooth_tessellation, n_surf_decimate, [('surface','in_file')]),
        (n_smooth_tessellation, n_surf_decimate,
         [(('surface',fname_presuffix_basename,'','_mask'), 'out_file')]),
        (n_surf_decimate, outputnode, [('out_file','surface')]),
        ])
    return w


def freesurfers2csv(aparc_stats,aseg_stats,meta_data):
    import numpy as np,os
    data = []
    md_keys = meta_data[0].keys()
    for parcs, seg, md  in zip(aparc_stats, aseg_stats, meta_data):
        lh = np.loadtxt(parcs[0], dtype=np.str)
        rh = np.loadtxt(parcs[1], dtype=np.str)
        aseg = np.loadtxt(seg, dtype=np.str)
        if not 'lh_labels' in locals():
            lh_labels = lh[:,0].tolist()
            rh_labels = rh[:,0].tolist()
            seg_labels = aseg[:,4].tolist()
        def fb_to_def(l,d,i):
            w=np.where(d[:,i]==l)[0]
            if w.size>0:
                return d[w[0]]
            else:
                return np.ones(d.shape[1])*-1
        lh_data = np.hstack([fb_to_def(l,lh,0)[2:5] for l in lh_labels])
        rh_data = np.hstack([fb_to_def(l,rh,0)[2:5] for l in rh_labels])
        seg_data = np.hstack([fb_to_def(l,aseg,4)[3] for l in seg_labels])
        meta = [md[k] for k in md_keys]
        data.append(np.hstack((meta,seg_data,lh_data,rh_data,)))
        if not 'header' in locals():
            header = np.hstack([md_keys+seg_labels] +[
                    reduce(lambda l,x: l+[x+'_%s_%s'%(hemi,meas) for meas in ['surf','vol','thick']], labels,[]) for hemi,labels in [('l',lh_labels),('r',rh_labels)]])
        del lh, rh, aseg

    out_fname = os.path.join(os.getcwd(),'freesurfer_stats.csv',)
    np.savetxt(out_fname, np.vstack(data), '"%s"', header=','.join(header.tolist()),delimiter=',')
    return out_fname
