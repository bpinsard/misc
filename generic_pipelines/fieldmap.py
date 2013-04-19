import os,operator,functools
from nipype.interfaces import spm, fsl, afni, utility, dcm2nii
import nipype.pipeline.engine as pe
import nipype.interfaces.io as nio
import nipype.interfaces.dcmstack as np_dcmstack
import nipype.utils.filemanip as fmanip
from .utils import *

EffEchoSpacing = (0x0043,0x102c)

def dicom_convert_ppl(name='b0fm_dicom_cvt',mapnode=False):
    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['dicom_files','out_file_format',
                    'voxel_order','meta_force_add']),
        name='inputspec')

    klass = pe.Node
    if mapnode:
        klass=pe.MapNode

    n_convert_b0fm_dicom = klass(
        np_dcmstack.DCMStackFieldmap(),
        iterfield='dicom_files',
        name='convert_b0fm_dicom')
    
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fieldmap_file','magnitude_file']),
        name='outputspec')

    w = pe.Workflow(name=name)
    w.connect([
       (inputnode, n_convert_b0fm_dicom,[('dicom_files','dicom_files'),
                                         ('out_file_format','out_file_format'),
                                         ('voxel_order','voxel_order'),
                                         ('meta_force_add','meta_force_add')]),
       (n_convert_b0fm_dicom, outputnode,[('fieldmap_file','fieldmap_file'),
                                          ('magnitude_file','magnitude_file')])
       ])
    return w


def fieldmap_prepare_files(converted_files,rwv_rescale_intercept=0,\
                               rwv_rescale_slope=1):
    if len(converted_files)==0:
        return '',''
    import nibabel as nb, numpy as np, os, re
    from nipype.utils.filemanip import fname_presuffix
    in_pattern='(?P<data>\d{8})_(?P<time>\d{6})(?P<site>\d{3})S(?P<subj>\d{4})'
    m = re.search(in_pattern,converted_files[0])
    out_file = './%(site)s_S_%(subj)s_%(data)s_%(time)s.nii.gz'%m.groupdict()
    niis = [nb.load(f) for f in converted_files]
    if not (rwv_rescale_intercept==0 and rwv_rescale_intercept==1):
        datas = [nii.get_data()*rwv_rescale_slope+rwv_rescale_intercept for nii in niis]    
    else:
        datas = [nii.get_data() for nii in niis]
    if len(datas)==2 and datas[0].ndim==4 and datas[0].shape[3]==2:
        #pair of complex data
        cplx1=datas[0][...,0]+1j*datas[0][...,1]
        cplx2=datas[1][...,0]+1j*datas[1][...,1]
    else:
        return
    phase_diff=np.mod(np.angle(cplx2)-np.angle(cplx1)+np.pi*2,np.pi*2)
    #append a zero image for FUGUE
    phase_diff=np.concatenate((phase_diff[...,np.newaxis],
                               np.zeros(phase_diff.shape+(1,))),3)
    phase_diff=phase_diff.astype(np.float32)
    mag1=np.abs(cplx1).astype(np.float32)
    phasediff_name=fname_presuffix(out_file,
                                   suffix='_phasediff',
                                   newpath=os.getcwd())
    mag_name=fname_presuffix(out_file,
                             suffix='_mag',
                             newpath=os.getcwd())
    nb.save(nb.Nifti1Image(phase_diff,niis[0].get_affine()), phasediff_name)
    nb.save(nb.Nifti1Image(mag1,niis[0].get_affine()), mag_name)
    return phasediff_name, mag_name


def fieldmap_extract_dicom_parameters(dicom_folder):
    if len(dicom_folder)==0:
        return
    import glob,os,dicom,numpy as np
    dicomfiles = glob.glob(os.path.join(dicom_folder,'*.dcm'))
    dcm_hdr = [dicom.read_file(f) for f in dicomfiles]
    tes=np.array([float(f.EchoTime) for f in dcm_hdr])
    te_long=tes.max()/1000
    te_short=tes.min()/1000
    delta_te = te_long-te_short
    #real-world value rescaling
    rwv_rescale_intercept = 0
    rwv_rescale_slope = 1
    if dcm_hdr[0].has_key((0x0040,0x9096)):
        rwv = dcm_hdr[0][0x0040,0x9096][0]
        if rwv.has_key((0x0040, 0x9224)) and rwv.has_key((0x0040, 0x9225)):
            rwv_rescale_intercept = float(rwv[0x0040,0x9224].value)
            rwv_rescale_slope = float(rwv[0x0040,0x9225].value)
    return te_short,te_long,delta_te, \
        rwv_rescale_intercept,rwv_rescale_slope

def convert_fieldmap_dicom(name='convert_fmap_dicoms'):
    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fieldmap_dicom_dir']),
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['phase_difference','magnitude','delta_TE']),
        name='outputspec')

    n_fieldmap_dicom_files=pe.Node(
        nio.DataGrabber(
            template='*.dcm',
            sort_filelist=True,),
        name='fieldmap_dicom_files')

    n_convert_b0_dicom = pe.Node(
        dcm2nii.Dcm2nii(
            gzip_output=True,
            id_in_filename=True,
            date_in_filename=True,
            config_file='/home_local/bpinsard/.dcm2nii/dcm2nii.ini'),
        name='convert_b0_dicom')

    n_extract_dicom_parameters = pe.Node(
        utility.Function(
            input_names=['dicom_folder'],
            output_names=['TE1','TE2','delta_TE',
                          'rwv_rescale_intercept','rwv_rescale_slope'],
            function=fieldmap_extract_dicom_parameters),
        name='extract_dicom_parameters')

    n_prepare_files = pe.Node(
        utility.Function(
            input_names=['converted_files',
                         'rwv_rescale_intercept','rwv_rescale_slope'],
            output_names=['phase_difference','magnitude'],
            function=fieldmap_prepare_files),
        name='prepare_files')

    w=pe.Workflow(name=name)
    w.connect([
      (inputnode,n_fieldmap_dicom_files,[('fieldmap_dicom_dir',
                                          'base_directory')]),
      (n_fieldmap_dicom_files,n_convert_b0_dicom,[('outfiles',
                                                   'source_names')]),
      (inputnode,n_extract_dicom_parameters,[('fieldmap_dicom_dir',
                                              'dicom_folder')]),
      (n_extract_dicom_parameters, n_prepare_files, [
                    ('rwv_rescale_intercept','rwv_rescale_intercept'),
                    ('rwv_rescale_slope','rwv_rescale_slope')]),
      (n_convert_b0_dicom,n_prepare_files,[('converted_files',
                                            'converted_files')]),
      (n_prepare_files, outputnode, [('phase_difference','phase_difference'),
                                     ('magnitude','magnitude')]),
      (n_extract_dicom_parameters, outputnode, [('delta_TE','delta_TE')]),
      ])
    return w


def make_fmap_wkfl_old(name='make_fieldmap'):

    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['phase_difference','magnitude',
                    't1_mask','t1_mag', 'delta_TE']),
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fieldmap','fieldmap_reg',
                    'fieldmap_magnitude','fieldmap_mask']),
        name='outputspec')
    
    n_fieldmap2t1_warp = pe.Node(
        fsl.FLIRT(out_matrix_file='fieldmap2t1.mat',
                  cost='normmi',
                  dof=6,
                  searchr_x=[-5,5], # restrict search as they are acquired 
                  searchr_y=[-5,5], # in the same sequence
                  searchr_z=[-5,5],
                  cost_func='normmi'),
        name='fieldmap2t1_warp')
    n_invert_fieldmap2t1_warp = pe.Node(
        fsl.ConvertXFM(invert_xfm=True),
        name='invert_fieldmap2t1_warp')
    n_warp_t1_mask = pe.Node(
        fsl.ApplyXfm(apply_xfm=True,interp='nearestneighbour'),
        name='warp_t1_mask')
    
    n_mask_mag = pe.Node(
        interface=fsl.ImageMaths(op_string='-mul', suffix='_brain',
                                 output_type='NIFTI'),
        name='mask_mag')
    
    n_unwrap_phasediff = pe.Node(
        fsl.PRELUDE(process3d=True),
        name='unwrap_phasediff')

    n_make_fieldmap = pe.Node(
        fsl.FUGUE(fmap_out_file='fieldmap.nii.gz',
                  smooth3d=2),
        name='make_fieldmap')

    w=pe.Workflow(name=name)
    w.connect([
      (inputnode,n_fieldmap2t1_warp,[('t1_mag','reference'),
                                     ('magnitude','in_file')]),
      (n_fieldmap2t1_warp,n_invert_fieldmap2t1_warp, [('out_matrix_file',
                                                       'in_file')]),

      (n_invert_fieldmap2t1_warp,n_warp_t1_mask, [('out_file',
                                                   'in_matrix_file')]),
      (inputnode, n_warp_t1_mask, [('t1_mask','in_file')]),
      (inputnode,n_warp_t1_mask,[('magnitude','reference')]),

      (inputnode,n_mask_mag,[('magnitude','in_file')]),
      (n_warp_t1_mask,n_mask_mag,[('out_file','in_file2')]),

      (inputnode,n_unwrap_phasediff,[('phase_difference','phase_file')]),
      (inputnode,n_unwrap_phasediff,[('magnitude','magnitude_file')]),
      (n_warp_t1_mask,n_unwrap_phasediff,[('out_file','mask_file')]),

      (inputnode,n_make_fieldmap,[
           (('phase_difference',fname_presuffix_basename,'fieldmap_','','./'),
            'fmap_out_file')]),
      (n_warp_t1_mask, n_make_fieldmap, [('out_file','mask_file')]),
      (n_unwrap_phasediff,n_make_fieldmap,[('unwrapped_phase_file',
                                            'phasemap_file')]),
      (inputnode,n_make_fieldmap,[('delta_TE', 'asym_se_time')]),

      (n_warp_t1_mask,outputnode,[('out_file','fieldmap_mask')]),
      (n_mask_mag,outputnode,[('out_file','fieldmap_magnitude')]),
      (n_make_fieldmap,outputnode,[('fmap_out_file','fieldmap')]),
      (n_make_fieldmap,outputnode,[('fmap_out_file','fieldmap_reg')]),
      ])

    return w 

def make_fmap_wkfl(name='make_fieldmap',mapnode=False):

    klass = pe.Node
    if mapnode:
        klass = pe.MapNode

    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fieldmap','magnitude',
                    't1_mask','t1_mag', 'delta_TE']),
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fieldmap','fieldmap_reg',
                    'fieldmap_magnitude','fieldmap_mask']),
        name='outputspec')
    
    n_fieldmap2t1_warp = klass(
        fsl.FLIRT(out_matrix_file='fieldmap2t1.mat',
                  cost='normmi',
                  dof=6,
                  searchr_x=[-5,5], # restrict search as they are acquired 
                  searchr_y=[-5,5], # in the same sequence
                  searchr_z=[-5,5],
                  cost_func='normmi'),
        iterfield = ['in_file'],
        name='fieldmap2t1_warp')
    n_invert_fieldmap2t1_warp = klass(
        fsl.ConvertXFM(invert_xfm=True),
        iterfield=['in_file'],
        name='invert_fieldmap2t1_warp')
    n_warp_t1_mask = klass(
        fsl.ApplyXfm(apply_xfm=True,interp='nearestneighbour',
                     datatype='char'),
        iterfield = ['in_matrix_file','reference'],
        name='warp_t1_mask')
    
    n_mask_mag = klass(
        fsl.ImageMaths(op_string='-mul', suffix='_brain',
                       output_type='NIFTI'),
        iterfield = ['in_file','in_file2'],
        name='mask_mag')
    
    n_make_fieldmap = klass(
        fsl.FUGUE(fmap_out_file='fieldmap.nii.gz',
                  smooth3d=2),
        iterfield = ['fmap_out_file','mask_file','fmap_in_file'],
        name='make_fieldmap')

    w=pe.Workflow(name=name)
    w.connect([
      (inputnode,n_fieldmap2t1_warp,[('t1_mag','reference'),
                                     ('magnitude','in_file')]),
      (n_fieldmap2t1_warp,n_invert_fieldmap2t1_warp, [('out_matrix_file',
                                                       'in_file')]),

      (n_invert_fieldmap2t1_warp,n_warp_t1_mask, [('out_file',
                                                   'in_matrix_file')]),
      (inputnode, n_warp_t1_mask, [('t1_mask','in_file')]),
      (inputnode,n_warp_t1_mask,[('magnitude','reference')]),

      (inputnode,n_mask_mag,[('magnitude','in_file')]),
      (n_warp_t1_mask,n_mask_mag,[('out_file','in_file2')]),

      (inputnode,n_make_fieldmap,[(('fieldmap',fname_presuffix_basename,'','_reg','./'),'fmap_out_file')]),
      (n_warp_t1_mask, n_make_fieldmap, [('out_file','mask_file')]),
      (inputnode,n_make_fieldmap,[('fieldmap','fmap_in_file')]),
      (inputnode,n_make_fieldmap,[('delta_TE', 'asym_se_time')]),

      (n_warp_t1_mask,outputnode,[('out_file','fieldmap_mask')]),
      (n_mask_mag,outputnode,[('out_file','fieldmap_magnitude')]),
      (n_make_fieldmap,outputnode,[('fmap_out_file','fieldmap')]),
      (n_make_fieldmap,outputnode,[('fmap_out_file','fieldmap_reg')]),
      ])

    return w 


def complex_to_mag_phase(complex_in_file,mask_file=None):
    import os
    import nibabel as nb
    import numpy as np
    from nipype.utils.filemanip import fname_presuffix
    cplx = nb.load(complex_in_file)
    data = cplx.get_data()
    mag_fname = fname_presuffix(complex_in_file,
                                suffix='_mag', newpath=os.getcwd())
    phase_fname = fname_presuffix(complex_in_file,
                                  suffix='_phase', newpath=os.getcwd())
    mag = np.abs(data)
    phases = np.angle(data)
    if mask_file != None:
        invmask = nb.load(mask_file).get_data() == 0
        mag[invmask] = 0
        phases[invmask] = 0
    nb.save(nb.Nifti1Image(mag,cplx.get_affine()),mag_fname)
    nb.save(nb.Nifti1Image(phases,cplx.get_affine()),phase_fname)
    return mag_fname, phase_fname
    

def flirt_complex(mat_file,cplx_file,ref_file):
    import numpy as np, os, nibabel as nb
    import nipy.algorithms.registration.groupwise_registration as gr
    from nipype.utils.filemanip import fname_presuffix
    mat = np.loadtxt(mat_file)
    ref = nb.load(ref_file)
    cplx = nb.load(cplx_file)
    grid=np.squeeze(np.mgrid[[slice(0,s) for s in ref.shape]+[slice(1,2)]])
    def _x_flipper(N_i):
        flipr = np.diag([-1, 1, 1, 1])
        flipr[0,3] = N_i - 1
        return flipr

    def flirt2aff(mat, in_img, ref_img):
        in_hdr = in_img.get_header()
        ref_hdr = ref_img.get_header()
        inspace = np.diag(in_hdr.get_zooms()[:3] + (1,))
        refspace = np.diag(ref_hdr.get_zooms()[:3] + (1,))
        if np.linalg.det(in_img.get_affine())>=0:
            inspace=np.dot(inspace, _x_flipper(in_hdr.get_data_shape()[0]))
        if np.linalg.det(ref_img.get_affine())>=0:
            refspace=np.dot(refspace, _x_flipper(ref_hdr.get_data_shape()[0]))
        return np.dot(np.linalg.inv(refspace), np.dot(mat, inspace))
    mat=flirt2aff(mat,cplx,ref)
    coords = np.linalg.inv(mat).dot(grid.transpose(1,2,0,3))
    tmp = np.zeros(ref.shape)
    out_cplx = np.zeros(ref.shape + cplx.shape[3:], dtype=np.complex64)
    for t in range((out_cplx.shape[3:]+(1,))[0]):
        splines = gr._cspline_transform(np.real(cplx.get_data()[...,t]))
        gr._cspline_sample3d(tmp,splines,coords[0],coords[1],coords[2])
        out_cplx[...,t] = tmp
        splines = gr._cspline_transform(np.imag(cplx.get_data()[...,t]))
        gr._cspline_sample3d(tmp,splines,coords[0],coords[1],coords[2])
        out_cplx[...,t] += 1j*tmp
    flirtname=fname_presuffix(cplx_file,suffix='_flirted',newpath=os.getcwd())
    nb.save(nb.Nifti1Image(out_cplx,ref.get_affine()),flirtname)
    return flirtname

def make_t1_fieldmap(name='make_t1_fieldmap'):
    

    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=['complex_image','magnitude_image',
                    't1_mask','t1_mag', 'delta_TE']),
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['fieldmap','fieldmap_mask']),
        name='outputspec')

    n_fieldmap_to_t1 = pe.Node(
        fsl.FLIRT(cost='mutualinfo',
                  dof=6,
                  no_search=True,
                  uses_qform=True),
        name='fieldmap_to_t1')

    n_flirt_complex = pe.Node(
        utility.Function(input_names=['mat_file','cplx_file','ref_file'],
                         output_names=['out_file'],
                         function=flirt_complex),
        name='flirt_complex')

    n_resample_complex = pe.Node(
        afni.Resample(resample_mode='Cu',out_file='%s_int1sp.nii.gz'),
        name='resample_complex')
    
    n_phase_mag = pe.Node(
        utility.Function(
            input_names=['complex_in_file', 'mask_file'],
            output_names=['magnitude_out_file','phase_out_file'],
            function = complex_to_mag_phase),
        name='phase_mag')

    n_unwrap_phases = pe.Node(
        fsl.PRELUDE(process3d=True),
        name='unwrap_phases')


    n_make_fieldmap = pe.Node(
        fsl.FUGUE(smooth3d=3),
        iterfield = ['fmap_out_file','mask_file','fmap_in_file'],
        name='make_fieldmap')

    w=pe.Workflow(name=name)

    w.connect([
       (inputnode, n_resample_complex,[('complex_image','in_file'),
                                       ('t1_mask','master')]),
       (inputnode, n_fieldmap_to_t1, [('magnitude_image','in_file'),
                                      ('t1_mag','reference')]),
       (n_fieldmap_to_t1, n_flirt_complex, [('out_matrix_file','mat_file'),]),
       (inputnode, n_flirt_complex, [('complex_image','cplx_file'),
                                     ('t1_mag','ref_file')]),
       (n_flirt_complex, n_phase_mag,[('out_file','complex_in_file')]),
       (inputnode ,n_phase_mag,[('t1_mask','mask_file')]),
       (n_phase_mag, n_unwrap_phases, [('phase_out_file','phase_file')]),
       (n_phase_mag,n_unwrap_phases,[('magnitude_out_file','magnitude_file')]),
       (inputnode, n_unwrap_phases, [('t1_mask','mask_file')]),
       (n_unwrap_phases,n_make_fieldmap,[
                    ('unwrapped_phase_file','phasemap_file')]),
       (inputnode,n_make_fieldmap,[
                    ('delta_TE','asym_se_time'),
                    ('t1_mask','mask_file'),
                    (('complex_image',fname_presuffix_basename,'','_fieldmap'),'fmap_out_file')]),
       ])

    return w
