from nipype.interfaces import fsl, afni, nitime, utility, freesurfer
import nipype.pipeline.engine as pe
import nipype.utils.filemanip as fmanip
from nipype.interfaces.base import Undefined



def pet_register_rois(name='pet_register_rois',pet_prefix='fdg'):
    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['pet_image','t1_image','rois_image']),
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['pet2t1','pet2t1_xfm','warped_rois','stats']),
        name='outputspec')

    n_flirt_pet2t1 = pe.Node(
        fsl.FLIRT(dof=6, cost='mutualinfo',
                  out_file=Undefined,
                  out_matrix_file='%spet2t1.mat'%pet_prefix),
        name='flirt_%spet2t1'%pet_prefix)

    n_fsl2xfm = pe.Node(
        freesurfer.preprocess.Tkregister(
            no_edit=True,xfm_out='%s.xfm',reg_file='%s.dat'),
        name='fsl2xfm')

    n_warp_rois = pe.Node(
        freesurfer.ApplyVolTransform(
            inverse=True, interp='nearest',
            transformed_file='warped_rois.nii.gz'),
        name='warp_rois')

    w=pe.Workflow(name=name)

    n_extract_signal = pe.Node(
        afni.ROIStats(args='-nzvoxels -nzsigma -nzmedian -nobriklab'),
        name='extract_signal')

    w.connect([
       (inputnode, n_flirt_pet2t1, [('pet_image','in_file'),
                                    ('t1_image','reference')]),
       (n_flirt_pet2t1, n_fsl2xfm, [('out_matrix_file','fsl_reg')]),
       (inputnode, n_fsl2xfm,[('pet_image','mov'),('t1_image','target'),]),
       (n_fsl2xfm, n_warp_rois,[('xfm_out','xfm_reg_file')]),
       (inputnode, n_warp_rois, [('pet_image','source_file'),
                                 ('rois_image','target_file')]),
       (n_warp_rois, n_extract_signal, [('transformed_file','mask')]),
       (inputnode, n_extract_signal, [('pet_image','in_file')]),
       (n_warp_rois, outputnode, [('transformed_file','warped_rois')]),
       (n_flirt_pet2t1, outputnode, [('out_matrix_file','pet2t1')]),
       (n_fsl2xfm, outputnode, [('xfm_out','pet2t1_xfm')]),
       (n_extract_signal, outputnode, [('stats','stats')])
       ])
    return w
