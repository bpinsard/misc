from nipype.interfaces import spm, fsl, afni, utility

import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.workflows.rsfmri.spm as spmrest
import nipype.interfaces.nipy.utils as nipyutils
import nipype.interfaces.nipy.preprocess as nipypp
import nipype.utils.filemanip as fmanip
from .utils import *

#convert t1 and crop to reduced box for algorithms efficiency
def t1_filename(dicom_files, ext):
    import dicom
    t1_filenames = []
    f=dicom_files[0]
    f=dicom.read_file(f)
    return '%s_%s_%s_%s_%s.%s'%tuple([f.PatientID, f.SeriesDate]+
            f.SeriesTime.split('.')+
            [f.SeriesDescription,ext])

def sort_t1_files(l):
    import numpy as np
    if isinstance(l,list):
        return [l[i] for i in np.array([int(f.split('_')[-3]) for f in l]).argsort()]
    else:
        return l


def t1_pipeline(name='t1_preproc'):
    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['t1_dicom_dir']),
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['mask','corrected_t1','corrected_t1_brain']),
        name='outputspec')


    n_t1_dicom_files=pe.Node(
        nio.DataGrabber(
            sort_filelist=True,),
        name='t1_dicom_files')

    n_to3d_t1 = pe.Node(
        afni.To3D(filetype='anat', environ=dict(AFNI_DICOM_RESCALE='YES')),
        name='to3d_t1')

    n_reorient_t1 = pe.Node(
        afni.Resample(orientation='RPI'),
        name='reorient_t1')

    n_autobox_t1 = pe.Node(
        afni.Autobox(padding=5),
        name='autobox_t1')

    n_zcut_t1 = pe.Node(
        afni.ZCutUp(outputtype='NIFTI'),
        name='zcut_t1')


    n_newsegment_t1 = pe.Node(
        spm.NewSegment(write_deformation_fields=[True,True],
                       channel_info = (0.0001,60,(True,True))),
        name='newsegment_t1')

    n_seg2mask = pe.Node(
        fsl.MultiImageMaths(
            output_type='NIFTI',
            op_string=' -add %s -add %s -thr 0.8 -bin -eroF -dilF -dilF'),
        name='seg2mask')

    n_mask_brain = pe.Node(
        interface=fsl.ImageMaths(op_string='-mul', suffix='_brain',
                                 output_type='NIFTI'),
        name='mask_brain')

    w = pe.Workflow(name=name)
    
    def zmax2keep(z):
        return '%d %d'%(max(0,z-174),z)

    w.connect([
        (inputnode, n_t1_dicom_files,[('t1_dicom_dir','base_directory')]),
        (n_t1_dicom_files, n_to3d_t1,[
                    (('outfiles', sort_t1_files),'in_files'),
                    (('outfiles', t1_filename,'nii.gz'), 'out_file')]),
        (n_to3d_t1, n_reorient_t1, [('out_file','in_file')]),
        (n_reorient_t1, n_autobox_t1, [('out_file','in_file')]),
        (n_reorient_t1, n_zcut_t1, [('out_file','in_file')]),
        (n_autobox_t1, n_zcut_t1, [(('z_max',zmax2keep),'keep')]),
        (n_zcut_t1, n_newsegment_t1, [('out_file','channel_files')]),
        (n_newsegment_t1, n_seg2mask, [
            (('native_class_images',getitem_rec,0,0), 'in_file'),
            (('native_class_images',getitem_rec,slice(1,3),0),'operand_files')]),
        (n_zcut_t1,n_seg2mask,[
            (('out_file',fname_presuffix_basename,'','_mask','.'),
             'out_file')]),
        (n_newsegment_t1, n_mask_brain, [('bias_corrected_images','in_file')]),
        (n_seg2mask, n_mask_brain, [('out_file','in_file2')]),
        (n_seg2mask, outputnode,[('out_file','mask')]),
        (n_newsegment_t1,outputnode,[('bias_corrected_images','corrected_t1')])
        
        ])
    return w
