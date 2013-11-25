import os
from nipype.interfaces import spm, fsl, afni, nitime, dcm2nii, utility, lif, freesurfer
 
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.workflows.rsfmri.spm as spmrest
import nipype.interfaces.nipy.utils as nipyutils
import nipype.interfaces.nipy.preprocess as nipypp
import nipype.algorithms.misc as algmisc
import nipype.interfaces.dcmstack as np_dcmstack
import nipype.utils.filemanip as fmanip
import nipype.pipeline.file_proxy as fileproxy
from .utils import *
from nipype.interfaces.base import Undefined

def get_dicom_par(dicom_pattern):
    import dicom, glob, struct
    #get the first dicom file found
    f = dicom.read_file(glob.glob(dicom_pattern)[0])
    effective_echo_spacing = 0
    try:
        effective_echo_spacing = float(f[0x0043,0x102c])
    except:
        try:
            wfs = float(f[0x2001,0x1022].value)
            if isinstance(wfs,str):
                wfs = struct.unpack('f',wfs)[0]
            effective_echo_spacing = (wfs)/(434.215 * (float(f.EchoTrainLength)+1))
        except:
            pass
    return dict(
        ID=f.PatientID,
        RepetitionTime=int(f.RepetitionTime)*0.001,
        SeriesDate=f.SeriesDate,
        SeriesTime=f.SeriesTime,
        StudyDate=f.StudyDate,
        StudyTime=f.StudyTime,
        SliceThickness=float(f.SliceThickness),
        PixelSpacing=f.PixelSpacing,
        EchoTime=float(f.EchoTime)*0.001,
        FlipAngle=int(f.FlipAngle),
        BirthDate=f.PatientsBirthDate,
        Sex=f.PatientsSex,
        Weight=float(f.PatientsWeight),
        Rows=f.Rows,
        Columns=f.Columns,
        EffectiveEchoSpacing=effective_echo_spacing)


def dicom_convert_ppl(name='fmri_dicom_cvt',mapnode=False):
    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['dicom_files','trim_dummy',
                    'out_file_format','meta_force_add',
                    'voxel_order']),
        name='inputspec')

    klass = pe.Node
    if mapnode:
        klass=pe.MapNode

    n_convert_fmri_dicom = klass(
        np_dcmstack.DCMStackfMRI(),
        iterfield='dicom_files',
        name='convert_fmri_dicom')

    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fmri_nifti']),
        name='outputspec')

    w=pe.Workflow(name=name)

    w.connect([
        (inputnode,n_convert_fmri_dicom,[('dicom_files','dicom_files'),
                                         ('trim_dummy','volume_range'),
                                         ('out_file_format','out_file_format'),
                                         ('voxel_order','voxel_order'),
                                         ('meta_force_add','meta_force_add')]),
        (n_convert_fmri_dicom,outputnode,[('nifti_file','fmri_nifti')])
        ])
    return w

def convert_rsfmri_dicom(name='convert_rsfmri_dicom'):
    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['rsfmri_dicom_pattern']),
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['rsfmri_file','dicom_pars']),
        name='outputspec')

    n_dcm2nii = pe.Node(
        dcm2nii.Dcm2nii(
            gzip_output=True,
            id_in_filename=True,
            date_in_filename=True,
            convert_all_pars=False,
            config_file='/home_local/bpinsard/.dcm2nii/dcm2nii.ini',),
        name='dcm2nii')

    n_rename_rsfmri = pe.Node(
        utility.Rename(
           format_string='%(site)s_S_%(subj)s_%(data)s_%(time)s_rsfMRI.nii.gz',
           parse_string='(?P<data>\d{8})_(?P<time>\d{6})(?P<site>\d{3})S(?P<subj>\d{4})'),
        name='rename_rsfmri')

    n_get_dicom_par=pe.Node(
        utility.Function(
            input_names=['dicom_pattern'],
            output_names=['dicom_pars'],
            function=get_dicom_par),
        name='get_dicom_par')

    w=pe.Workflow(name=name)
    w.connect([
        (inputnode, n_dcm2nii, [('rsfmri_dicom_pattern','source_names')]),
        (n_dcm2nii, n_rename_rsfmri,[('converted_files','in_file')]),
        (inputnode,n_get_dicom_par,[('rsfmri_dicom_pattern','dicom_pattern')]),
        (n_rename_rsfmri, outputnode, [('out_file','rsfmri_file')]),
        (n_get_dicom_par, outputnode, [('dicom_pars','dicom_pars')]),
        ])
    return w

def fmri_qc(name='fmri_qc'):

    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['motion','realigned','mask','grey']),
        name='inputspec')
    
    n_tsnr = pe.Node(algmisc.TSNR(), name='tsnr')

    def tsnr_stats(tsnr,mask,grey):
        import nibabel as nb, numpy as np
        tsnr=nb.load(tsnr).get_data()
        mask=np.logical_and(
            nb.load(mask).get_data()>0,
            np.logical_not(np.logical_or(np.isinf(tsnr),np.isnan(tsnr))))
        grey=np.logical_and(nb.load(grey).get_data()>0, mask)
        brain_tsnr_mean = tsnr[mask].mean()
        brain_tsnr_std = tsnr[mask].std()
        grey_tsnr_mean = tsnr[grey].mean()
        grey_tsnr_std = tsnr[grey].std()
        return brain_tsnr_mean,brain_tsnr_std,grey_tsnr_mean,grey_tsnr_std
    
    n_tsnr_stats = pe.Node(
        utility.Function(
            input_names=['tsnr','mask','grey'],
            output_names=['brain_tsnr_mean','brain_tsnr_std',
                          'grey_tsnr_mean','grey_tsnr_std'],
            function=tsnr_stats),
        name='tsnr_stats')

    w=pe.Workflow(name=name)
    w.connect([
            (inputnode,n_tsnr,[('realigned','in_file')]),
#            (n_tsnr,n_tsnr_stats,[('tsnr_file','tsnr')]),
#            (inputnode,n_tsnr_stats,[('mask','mask'),
#                                     ('grey','grey')]),

            ])
    return w

def epi_correction(name='epi_correction'):
    
    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fieldmap','fieldmap_mag','fieldmap_mask',
                    'epi_file','epi_reference','epi_mask',
                    'echo_time','echo_spacing','unwarp_direction']),
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['unwarped_epi','voxel_shift_map',
                    'fieldmap2epi','epi2fieldmap',
                    'fieldmap','epi_sigloss']),
        name='outputspec')


    #compute deformation/signal_loss in fieldmap space to warp magnitude for registration
    #this has to use the epi parameters?
    n_signal_loss = pe.Node(
        fsl.SigLoss(),
        name='signal_loss')

    n_fieldmap_mag_lossy = pe.Node(
        interface=fsl.ImageMaths(op_string='-mul', suffix='_lossy',
                                 output_type='NIFTI'),
        name='fieldmap_mag_lossy')

    n_fm_voxelshiftmap = pe.Node(
        fsl.FUGUE(forward_warping=True,
                  nokspace=True,),
        name='fm_voxelshiftmap')
    #register fieldmap into EPI space
    n_estimate_warp = pe.Node(
        fsl.FLIRT(cost='normmi',
                  cost_func='normmi',
                  out_matrix_file='epi_to_b0fm',
                  interp='trilinear',
                  searchr_x=[-5,5], # restrict search as they are acquired 
                  searchr_y=[-5,5], # in the same sequence
                  searchr_z=[-5,5],
                  dof=6),
        name='estimate_warp')
    n_invert_warp = pe.Node(
        fsl.ConvertXFM(invert_xfm=True),
        name='invert_warp')
    n_warp_fieldmap = pe.Node(
        fsl.ApplyXfm(apply_xfm=True),
        name='warp_fieldmap')

    n_warp_sigloss = pe.Node(
        fsl.ApplyXfm(apply_xfm=True),
        name='warp_sigloss')
    
    #compute deformation in EPI space
    n_epi_voxelshiftmap = pe.Node(
        fsl.FUGUE(shift_out_file='vsm_epi.nii.gz',),
        name='epi_voxelshiftmap')

    n_unwarp_epi = pe.Node(
        fsl.FUGUE(),
        name='unwarp_epi')

    w=pe.Workflow(name=name)
    w.connect([
        (inputnode,n_signal_loss, [('fieldmap','in_file'),
                                   ('fieldmap_mask','mask_file'),
                                   ('echo_time','echo_time')]),
        (n_signal_loss, n_fieldmap_mag_lossy, [('out_file','in_file2')]),
        (inputnode, n_fieldmap_mag_lossy, [('fieldmap_mag','in_file')]),
        (inputnode,n_fm_voxelshiftmap,[('fieldmap','fmap_in_file'),
                                       ('echo_spacing','dwell_time'),
                                       ('unwarp_direction','unwarp_direction'),
                                       ('fieldmap_mask','mask_file')]),
        (n_fieldmap_mag_lossy, n_fm_voxelshiftmap, [('out_file','in_file')]),
        (n_signal_loss, n_estimate_warp, [('out_file','ref_weight')]),
        (n_fm_voxelshiftmap,n_estimate_warp,[('warped_file','reference')]),
        (inputnode,n_estimate_warp,[('epi_reference','in_file'),  ]),
                                    #('epi_mask','in_weight')]),
        (n_estimate_warp,n_invert_warp, [('out_matrix_file','in_file')]),
        
        (n_invert_warp,n_warp_fieldmap, [('out_file','in_matrix_file')]),
        (inputnode,n_warp_fieldmap, [('fieldmap','in_file')]),
        (inputnode,n_warp_fieldmap, [('epi_reference','reference')]),

        (n_invert_warp,n_warp_sigloss, [('out_file','in_matrix_file')]),
        (n_signal_loss,n_warp_sigloss, [('out_file','in_file')]),
        (inputnode,n_warp_sigloss, [('epi_reference','reference')]),
        
        (n_warp_fieldmap,n_epi_voxelshiftmap,[('out_file','fmap_in_file')]),
        (inputnode, n_epi_voxelshiftmap,[
                    ('epi_reference','in_file'),
                    ('epi_mask','mask_file'),
                    ('unwarp_direction','unwarp_direction'),
                    ('echo_spacing','dwell_time')]),
        (n_epi_voxelshiftmap,n_unwarp_epi,[('shift_out_file','shift_in_file')]),
        (inputnode,n_unwarp_epi,[
           ('epi_file','in_file'),
           ('epi_mask','mask_file')]),

        (n_estimate_warp,outputnode, [('out_matrix_file','epi2fieldmap')]),
        (n_invert_warp, outputnode, [('out_file','fieldmap2epi')]),
        (n_warp_sigloss,outputnode, [('out_file','epi_sigloss')]),
        (n_warp_fieldmap,outputnode,[('out_file','fieldmap')]),
        (n_unwarp_epi,outputnode,[('unwarped_file','unwarped_epi')]),
        (n_epi_voxelshiftmap,outputnode,[('shift_out_file','voxel_shift_map')]),
        ])
    return w

def n_volumes(f,ofst=0):
    import nibabel as nb
    return nb.load(f).shape[-1]+ofst

def base_preproc(trim_realign=True,name='rsfmri_base_preproc'):

    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fmri','t1','t1_mask']),
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['preprocessed','mask','mean','motion']),
        name='outputspec')

#    n_trim = pe.Node(
#        interface=nipypp.Trim(begin_index=3),
#        name='trim')
    
    n_realign = pe.Node(
        fsl.MCFLIRT(ref_vol=0,
                    save_plots=True,
                    save_rms=True,
                    save_mats=True,
                    stats_imgs=True,),
        name='realign')

    n_mean = pe.Node(fsl.MeanImage(),name='mean')

    n_mask = pe.Node(
        interface=afni.Automask(
            out_file='%s_mask.nii',
#            brain_file=Undefined,
            outputtype='NIFTI'),
        name='mask')

    n_mask_mean = pe.Node(
        interface=fsl.ImageMaths(op_string='-mul', suffix='_brain',
                                 output_type='NIFTI'),
        name='mask_mean')

    n_segment_epi = pe.Node(
        fsl.FAST(
            img_type=2,
            number_classes=3,
            probability_maps=True,
            segments=True),
        name='segment_epi')

    #linear with shear/scale in phase direction
    n_coregister_linear = pe.Node(
        afni.Allineate(epi=True, args='-float',cost='nmi',
                       out_param_file='params.1D',
                       out_matrix='coregister.mat'),
        name='coregister_linear')

    n_coregister_gray_linear = pe.Node(
        afni.Allineate(epi=True, args='-float',cost='nmi',
                       out_param_file='params.1D',
                       out_matrix='coregister.mat'),
        name='coregister_gray_linear')

    n_smooth = pe.Node(
        afni.BlurInMask(fwhm=5.0, out_file='%s_smooth', float_out=True),
        name = 'smooth')    

    n_bandpass_smooth = pe.Node(
        afni.Bandpass(highpass=0.005, lowpass=999999,
                      despike=True,
                      blur=5.0, normalize=False, out_file='%s_filt.nii.gz'),
        name='bandpass_smooth')

    n_motion_filter = pe.Node(
        interface = nipypp.RegressOutMotion(
            motion_source='fsl',
            regressors_type='voxelwise_translation',
            global_signal = False,
            prefix = 'mfilt_',
            regressors_transform='original+bw_derivatives'),
        name = 'motion_filter')


#    spm_path = spm.Info().version()['path']
#    epi_tpl = os.path.join(spm_path, 'templates/EPI.nii')
    """
    n_normalize = pe.Node(
        spm.Normalize(template=epi_tpl,
                      source_image_smoothing=8,
                      template_image_smoothing=0,
                      DCT_period_cutoff=25,
                      affine_regularization_type='mni',
                      jobtype='est'),
        name='normalize')
        """

    n_motion_estimates = pe.Node(
        nipyutils.MotionEstimate(motion_source='fsl'),
        name='motion_estimates')    

    w=pe.Workflow(name=name)

    if trim_realign:
        w.connect([
#                (inputnode, n_trim, [('fmri','in_file')]),
#                (inputnode, n_trim, [('fmri','in_file_a'),
#                                     (('fmri',n_volumes,-1),'stop_idx')]),
#                (n_trim, n_realign, [('out_file','in_file')]),
                (inputnode, n_realign, [('fmri','in_file')]),
#                (inputnode, n_realign, [('fmri','in_file')]),
                (n_realign, n_motion_filter, [('out_file','in_file'),
                                              ('par_file','motion')]),
                (n_mask, n_motion_filter,[('out_file','mask')]),
                (n_motion_filter, n_bandpass_smooth, [('out_file','in_file')]),
                (n_realign, n_mask, [('out_file','in_file')]),                
                (n_realign, n_mask_mean,  [('mean_img', 'in_file')]),
                (n_realign, n_motion_estimates,[('par_file','motion')]),
                (n_mask, n_motion_estimates,[('out_file','mask')]),
                (n_realign, outputnode, [('par_file','motion')]),
                ])
    else:
        w.connect([
                (inputnode, n_bandpass_smooth, [('fmri','in_file')]),
                (inputnode, n_mean, [('fmri','in_file')]),
                (inputnode, n_mask, [('fmri','in_file')]),                
                (n_mean, n_mask_mean, [('out_file', 'in_file')]),
                ])

    w.connect([
        (n_mask, n_mask_mean,  [('out_file', 'in_file2')]),
        (n_mask, n_bandpass_smooth, [('out_file','mask')]),
#        (n_mask_mean, n_segment_epi, [('out_file','in_files')]),
#        (n_mask_mean, n_normalize, [('out_file','source')]),
        
#        (n_detrend, n_smooth, [('out_file','in_file')]),
#        (n_mask, n_smooth,  [('out_file', 'mask')]),


#        (n_smooth, outputnode, [('out_file','preprocessed')]),
        (n_bandpass_smooth, outputnode, [('out_file','preprocessed')]),
        (n_mask, outputnode, [('out_file','mask')]),
        (n_mask_mean, outputnode, [('out_file','mean')]),

      ])
    return w


#first tentative for optimal registration of epi to t1 and MNI
def epi_normalize(name='epi_normalize'):

    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fmri_mean','t1','t1_to_mni']),
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['epi2t1_warp','coregistered_fmri_mean','epi2mni_warp',
                    't1_to_epi_warp']),
        name='outputspec')

    n_spm_coregister = pe.Node(
        spm.Coregister(jobtype='estimate'),
        name='spm_coregister')

    n_epi2mni = pe.Node(
        spm.preprocess.ApplyDeformations(
            reference_volume='/coconut/applis/src/spm8_x64/toolbox/Seg/TPM.nii'),
        name='epi2mni')


    n_flirt_epi2t1 = pe.Node(
        fsl.FLIRT(out_matrix_file='flirt_epi2t1.mat',
                  out_file='%.26s_flirt',
                  cost='normmi', # as in fcon1000 preproc, why??
                  searchr_x=[-10,10],searchr_y=[-10,10],searchr_z=[-10,10],
                  dof=6),
        name='flirt_epi2t1')
    n_t1_to_epi = pe.Node(
        fsl.ConvertXFM(invert_xfm=True),
        name='t1_to_epi')
    
    w=pe.Workflow(name=name)

    w.connect([
#        (inputnode,n_spm_coregister,[('fmri_mean','source'),
#                                     ('t1','target')]),
        (inputnode,n_flirt_epi2t1,[('t1','reference')]),
#        (n_spm_coregister,n_epi2mni,[('coregistered_source','in_files')]),
#        (inputnode,n_epi2mni,[('t1_to_mni','deformation_field')]),
        (inputnode,n_flirt_epi2t1,[('fmri_mean','in_file')]),
        (n_flirt_epi2t1,outputnode,[('out_matrix_file','epi2t1_warp')]),
        (n_flirt_epi2t1,n_t1_to_epi,[('out_matrix_file','in_file')]),
        (n_t1_to_epi,outputnode,[('out_file','t1_to_epi_warp')]),
#        (n_spm_coregister,outputnode,[('coregistered_source',
#                                       'coregistered_fmri_mean')]),
        ])
    return w

def restrict_to_gray(rois, mask, threshold=.5, min_nvox=12):
    import os, nibabel as nb, numpy as np
    from nipype.utils.filemanip import fname_presuffix
    if not isinstance(rois,list):
        rois = [rois]
    roi_niis = [nb.load(r) for r in rois]
    mask = nb.load(mask).get_data() > threshold
    rois_data = [r.get_data() for r in roi_niis]
    new_rois = [r*mask for r in rois_data]
    nfnames=[]
    for od,nd,nii,fname in zip(rois_data,new_rois,roi_niis,rois):
        for rid in np.unique(od)[1:]:
            if np.count_nonzero(nd==rid) < min_nvox:
                nd[od==rid] = rid
        nfname = fname_presuffix(fname,newpath=os.getcwd(),suffix='_gmonly')
        nb.save(nb.Nifti1Image(nd,nii.get_affine(),nii.get_header()),nfname)
        nfnames.append(nfname)
    return nfnames

def warp_rois_gray(name='warp_rois_gray'):

    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fmri_reference','gray_matter','t1_to_mni','rois_files',
                    't1_to_epi']),
        name='inputspec')

    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['t1_rois','t1_gray_rois','fmri_rois']),
        name='outputspec')
    
    n_mni_to_t1 = fileproxy.GunzipNode(
        spm.ApplyInverseDeformation(interpolation=0),
        name='mni_to_t1')

    n_restrict_to_gray = pe.Node(
        utility.Function(input_names=['rois','mask','threshold','min_nvox'],
                         output_names=['masked_rois'],
                         function=restrict_to_gray),
        name='restrict_to_gray')
    n_restrict_to_gray.inputs.min_nvox = 100
    n_restrict_to_gray.inputs.threshold = 1e-3

    n_t1_to_fmri = pe.MapNode(
        fsl.FLIRT(interp='nearestneighbour',
                  out_file='%s_epi',
                  apply_xfm=True,),
        iterfield=['in_file'],
        name='t1_to_fmri')

    w=pe.Workflow(name=name)
    w.connect([
            (inputnode,n_mni_to_t1,[('rois_files','in_files'),
                                    ('t1_to_mni','deformation_field'),
                                    ('gray_matter','target')]),
            (inputnode,n_restrict_to_gray,[('gray_matter','mask')]),
            (n_mni_to_t1,n_restrict_to_gray,[('out_files','rois')]),
            (n_restrict_to_gray,n_t1_to_fmri,[('masked_rois','in_file')]),
            (inputnode,n_t1_to_fmri,[('fmri_reference','reference'),
                                     ('t1_to_epi','in_matrix_file')]),
            (n_mni_to_t1,outputnode,[('out_files','t1_rois')]),
            (n_restrict_to_gray,outputnode,[('masked_rois','t1_gray_rois')]),
            (n_t1_to_fmri,outputnode,[('out_file','fmri_rois')]),
            ])
    return w

def bbr_coregister(name='bbr_coregister', use_fieldmap=True):
    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fmri_mean','t1','t1_mask','t1_to_mni', 'white_matter',
                    'fieldmap_to_t1', 'fieldmap','echospacing','pedir',
                    'init']),
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['epi2t1_warp','coregistered_fmri_mean']),
        name='outputspec')

    if use_fieldmap:
        n_warp_fieldmap2t1 = pe.Node(
            fsl.FLIRT(apply_xfm=True,),
            name='warp_fieldmap2t1')
    
    n_flirt_bbr=pe.Node(
        fsl.FLIRT(
            cost='bbr',cost_func='bbr',dof=6,
            output_type='NIFTI',
            schedule=os.path.join(os.environ['FSLDIR'],'etc','flirtsch','bbr.sch'),),
        name='flirt_bbr')
    
    n_flirt_nofmap = pe.Node(
        fsl.FLIRT(apply_xfm=True,),
        name='flirt_nofmap')


    n_epi2mni = fileproxy.GunzipNode(
        spm.preprocess.ApplyDeformations(
            reference_volume='/coconut/applis/src/spm8_x64/toolbox/Seg/TPM.nii'),
        name='epi2mni')

    w=pe.Workflow(name=name)

    w.connect([
      (inputnode,n_flirt_bbr,[('fmri_mean','in_file'),
                              ('t1','reference'),
                              ('white_matter','wm_seg'),
                              ('init','in_matrix_file')]),
      (n_flirt_bbr, outputnode, [('out_matrix_file','epi2t1_warp'),
                                 ('out_file','coregistered_fmri_mean')]),
      (n_flirt_bbr,n_epi2mni,[('out_file','in_files')]),
      (inputnode,n_epi2mni,[('t1_to_mni','deformation_field')]),

      ])
    if use_fieldmap:        
        w.connect([
                (inputnode,n_warp_fieldmap2t1,[
                        ('fieldmap','in_file'),
                        ('t1','reference'),
                        ('fieldmap_to_t1','in_matrix_file')]),
                (n_warp_fieldmap2t1,n_flirt_bbr,[('out_file','fieldmap')]),
                (inputnode,n_flirt_bbr,[
                        ('t1_mask','fieldmapmask'),
                        ('echospacing','echospacing'),
                        ('pedir','pedir')]),
#      (n_flirt_bbr, n_flirt_nofmap,[('out_matrix_file','in_matrix_file')]),
#      (inputnode, n_flirt_nofmap,[('t1','reference'),
#                                  ('fmri_mean','in_file')]),

                ])


    return w




def epi_unwrap(name='epi_unwrap'):

    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fieldmap', 'fieldmap_mask', 'fieldmap_to_epi',
                    'epi_file','epi_reference',
                    'echo_time','echo_spacing','unwarp_direction']),
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['unwarped_epi','voxel_shift_map']),
        name='outputspec')
    
    n_fieldmap_to_epi = pe.Node(
        fsl.FLIRT(apply_xfm=True,
                  no_resample_blur=True),
        name='fieldmap_to_epi')

    n_mask_to_epi = pe.Node(
        fsl.FLIRT(apply_xfm=True,
                  interp='nearestneighbour'),
        name='mask_to_epi')
    

    n_epi_voxelshiftmap = pe.Node(
        fsl.FUGUE(shift_out_file='vsm_epi.nii.gz',),
        name='epi_voxelshiftmap')

    n_unwarp_epi = pe.Node(
        fsl.FUGUE(),
        name='unwarp_epi')


    w=pe.Workflow(name=name)

    w.connect([
       (inputnode, n_fieldmap_to_epi, [
                    ('fieldmap_to_epi','in_matrix_file'),
                    ('fieldmap','in_file'),
                    ('epi_reference','reference'),]),
       (inputnode, n_mask_to_epi, [
                    ('fieldmap_to_epi','in_matrix_file'),
                    ('fieldmap_mask','in_file'),
                    ('epi_reference','reference'),]),
       
       (n_fieldmap_to_epi,n_epi_voxelshiftmap,[('out_file','fmap_in_file')]),
       (n_mask_to_epi, n_epi_voxelshiftmap, [('out_file','mask_file')]),
       (inputnode, n_epi_voxelshiftmap,[
                    ('epi_reference','in_file'),
                    ('unwarp_direction','unwarp_direction'),
                    ('echo_spacing','dwell_time')]),
       (n_mask_to_epi, n_unwarp_epi, [('out_file','mask_file')]),
       (n_epi_voxelshiftmap,n_unwarp_epi,[('shift_out_file','shift_in_file')]),
       (inputnode,n_unwarp_epi,[
                    ('epi_file','in_file'),]),
       ])
    
    return w



def extract_bbox(fname):
    nii = nb.load(fname)
    bbox = nii.get_affine().dot([[0]*3,nii.shape]).ravel().tolist()
    del nii
    return bbox

def spm_realign_opt(name='spm_realign_opt'):

    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fmri','reference','mask']),
        name='inputspec')

    outputnode = pe.Node(
        utility.IdentityInterface(fields=['motion_parameters','out_file']),
        name='outputspec')

    n_1st_vol = pe.Node(
        nipypp.Trim(end_index=1,out_file='%s_1vol.nii',outputtype='NIFTI'),
        name='1st_vol')

    n_spm_coregister = pe.Node(
        spm.Coregister(jobtype='estimate',cost_function='mi'),
        name='coregister')

    n_merge2realign = pe.Node(utility.Merge(2),
                              name='merge2realign')

    n_spm_realign = fileproxy.GunzipNode(
        spm.Realign(jobtype='estwrite', write_which=[2,1]),
        name='realign', proxy_out=False)

    n_resample_mask = pe.Node(
        afni.Resample(out_file='%s_epi.nii',outputtype='NIFTI'),
        name='resample_mask')
    
    def trim_mp(motion_parameters,range):
        import os,numpy as np
        out_file=os.path.abspath("./"+os.path.basename(motion_parameters))
        np.savetxt(out_file,np.loadtxt(motion_parameters)[range])
        return out_file

    n_trim_motion_pars=pe.Node(
        utility.Function(
            input_names=['motion_parameters','range'],
            output_names=['motion_parameters'],
            function=trim_mp),
        name='trim_motion_pars')
    n_trim_motion_pars.inputs.range=slice(1,None)


    w=pe.Workflow(name=name)
    w.connect([
            (inputnode, n_1st_vol, [('fmri','in_file')]),
            (inputnode, n_spm_coregister, [('reference','target')]),
            (n_1st_vol, n_spm_coregister, [('out_file','source')]),
            (n_spm_coregister,n_merge2realign,[('coregistered_source','in1')]),
            (inputnode, n_merge2realign, [('fmri','in2')]),
            (n_merge2realign, n_spm_realign, [('out','in_files')]),
            (n_spm_realign, n_resample_mask,[('mean_image','master')]),
            (inputnode, n_resample_mask,[('mask','in_file')]),

            (n_spm_realign,outputnode,
             [(('realigned_files',utility.select,1),'out_file'),]),
            (n_spm_realign, n_trim_motion_pars,[
                    ('realignment_parameters','motion_parameters')]),
            (n_trim_motion_pars,outputnode,[
                    ('motion_parameters','motion_parameters')]),
            ])
    return w


def fsl_realign_opt(name='fsl_realign_opt'):

    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fmri','reference','mask']),
        name='inputspec')

    n_flirt_epi2t1 = pe.Node(
        fsl.FLIRT(out_matrix_file='flirt_epi2t1.mat',
                  cost='normmi', # as in fcon1000 preproc, why??
                  searchr_x=[-10,10],searchr_y=[-10,10],searchr_z=[-10,10],
                  dof=6),
        name='flirt_epi2t1')
    
    n_mcflirt_spline = pe.Node(
        fsl.MCFLIRT(interpolation='spline',
                    ref_vol=0,
                    save_plots=True,
                    save_rms=True,
                    save_mats=True,
                    stats_imgs=True,
                    dof=6),
        name='mcflirt_spline')

    w=pe.Workflow(name=name)
    w.connect([
            (inputnode,n_flirt_epi2t1,[('fmri','in_file'),
                                       ('reference','reference')]),
            (inputnode,n_mcflirt_spline,[('fmri','in_file'),]),
            (n_flirt_epi2t1,n_mcflirt_spline,[('out_matrix_file','init')])
            ])
    return w


def epi_fs_coregister(name='epi_fs_coregister'):
    
    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fmri','subject_id','subjects_dir',
                    'roi_file','mask_file']),
        name='inputspec')

    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fmri_mask','fmri_rois','reg_file','fsl_reg_file']),
        name='outputspec')


    n_bbregister = pe.Node(
        freesurfer.BBRegister(init='fsl', contrast_type='t2',
                              out_fsl_file=True),
        name='bbregister')

    n_fsmask2epi = pe.Node(
        freesurfer.ApplyVolTransform(inverse=True, interp='nearest',
                                     transformed_file='mask_epi.nii.gz'),
        name='fsmask2epi')

    n_fsrois2epi = pe.Node(
        freesurfer.ApplyVolTransform(inverse=True, interp='nearest',
                                     transformed_file='epi_aparc.aseg.nii.gz'),
        name='fsrois2epi')
    w=pe.Workflow(name=name)
    w.connect([
            (inputnode, n_bbregister,[('fmri','source_file'),
                                      ('subjects_dir','subjects_dir'),
                                      ('subject_id','subject_id')]),
            (n_bbregister, n_fsrois2epi,[('out_reg_file','reg_file')]),
            (inputnode, n_fsrois2epi, [('fmri','source_file'),
                                       ('roi_file','target_file')]),
            (n_bbregister, n_fsmask2epi,[('out_reg_file','reg_file')]),
            (inputnode, n_fsmask2epi, [('fmri','source_file'),
                                       ('mask_file','target_file')]),
            (n_bbregister, outputnode,[('out_reg_file','reg_file')]),
            (n_bbregister, outputnode,[('out_fsl_file','fsl_reg_file')]),
            (n_fsmask2epi, outputnode, [('transformed_file','fmri_mask')]),
            (n_fsrois2epi, outputnode, [('transformed_file','fmri_rois')]),
            ])
    return w


def restrict_to_gray(rois, seg, tissues, min_nvox=12):
    import os, nibabel as nb, numpy as np
    from nipype.utils.filemanip import fname_presuffix
    if not isinstance(rois,list):
        rois = [rois]
    roi_niis = [nb.load(r) for r in rois]
    seg = nb.load(seg).get_data()
    mask = np.zeros(seg.shape,dtype=np.bool)
    for t in tissues:
        np.logical_or(mask,mask==t,mask)
    rois_data = [r.get_data() for r in roi_niis]
    new_rois = [r*mask for r in rois_data]
    nfnames=[]
    for od,nd,nii,fname in zip(rois_data,new_rois,roi_niis,rois):
        for rid in np.unique(od)[1:]:
            if np.count_nonzero(nd==rid) < min_nvox:
                nd[od==rid] = rid
        nfname = fname_presuffix(fname,newpath=os.getcwd(),suffix='_gmonly')
        nb.save(nb.Nifti1Image(nd,nii.get_affine(),nii.get_header()),nfname)
        nfnames.append(nfname)
    return nfnames


def warp_rois_gray_fs(name='warp_rois_gray_fs'):

    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fmri_reference','seg','mni_reg','def_field',
                    'rois_files','t1_to_epi']),
        name='inputspec')

    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['t1_rois','t1_gray_rois','fmri_rois']),
        name='outputspec')
    
    n_mni_to_t1 = pe.MapNode(
        freesurfer.ApplyVolTransform(interp='nearest',
                                     no_def_m3z_path=True,
                                     inverse=True,invert_morph=True),
        iterfield = ['target_file','transformed_file'],
        name='mni_to_t1')

    n_restrict_to_gray_fs = pe.Node(
        utility.Function(input_names=['rois','seg','class'],
                         output_names=['masked_rois'],
                         function=restrict_to_gray_fs),
        name='restrict_to_gray_fs')
    n_restrict_to_gray.inputs.min_nvox = 100
    n_restrict_to_gray.inputs.tissues = [2,42,8,47,12,51,11,50,10,49,18,54,26,58]
    n_restrict_to_gray.inputs.threshold = 1e-3

    n_t1_to_fmri = pe.MapNode(
        fsl.FLIRT(interp='nearestneighbour',
                  out_file='%s_epi',
                  apply_xfm=True,),
        iterfield=['in_file'],
        name='t1_to_fmri')

    w=pe.Workflow(name=name)
    w.connect([
            (inputnode,n_mni_to_t1,
             [('rois_files','target_file'),
              ((fname_presuffix,'rois_files','','_native'),'transformed_file'),
              ('def_field','m3z_file'),
              ('mni_reg','reg_file'),
              ('gray_matter','mov')]),
            (inputnode,n_restrict_to_gray,[('seg','tissues')]),
            (n_mni_to_t1,n_restrict_to_gray,[('transformed_file','rois')]),
            (n_restrict_to_gray,n_t1_to_fmri,[('masked_rois','in_file')]),
            (inputnode,n_t1_to_fmri,[('fmri_reference','reference'),
                                     ('t1_to_epi','in_matrix_file')]),
            (n_mni_to_t1,outputnode,[('out_files','t1_rois')]),
            (n_restrict_to_gray,outputnode,[('masked_rois','t1_gray_rois')]),
            (n_t1_to_fmri,outputnode,[('out_file','fmri_rois')]),
            ])
    return w
