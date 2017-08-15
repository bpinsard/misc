
import os,sys,glob,datetime
import numpy as np

from nipype.interfaces import spm, fsl, afni, nitime, utility, dcmstack as np_dcmstack, freesurfer, nipy, io as nio, ants

import nipype.pipeline.engine as pe

sys.path.insert(0,'/data/code/')
import generic_pipelines
from generic_pipelines.utils import wrap, fname_presuffix_basename

from nipype.interfaces.base import (TraitedSpec, BaseInterface, traits,
                    BaseInterfaceInputSpec, isdefined, File, Directory,
                    InputMultiPath, OutputMultiPath)
from nipype.utils.filemanip import filename_to_list

class GrayOrdinatesBandPassSmoothInputSpec(BaseInterfaceInputSpec):
    in_file = File(
        exists=True, mandatory=True,
        desc="input hdf5 timeseries file")
    smoothing_steps = traits.Int(desc="smoothing extent")
    smoothing_factor = traits.Float()
    data_field = traits.Str(
        'FMRI/DATA', usedefault=True,
        desc='the path of data to smooth in hdf5 file')
    filter_range = traits.Tuple(
        traits.Float, traits.Float, mandatory = True,
        desc='the upper and lower pass thresholds')
    TR = traits.Float(
        desc='override the TR in dataset')

    std_threshold = traits.Float(
        1.96, usedefault=True,
        desc="""determine the threshold of standard deviation to exclude 
bad voxels, which are reinterpolated from neighbouring voxels""")

class GrayOrdinatesBandPassSmoothOutputSpec(TraitedSpec):
    out_file = File(
        exists=True,
        desc='output hdf5 timeseries file')


class GrayOrdinatesBandPassSmooth(BaseInterface):
    input_spec = GrayOrdinatesBandPassSmoothInputSpec
    output_spec = GrayOrdinatesBandPassSmoothOutputSpec

    def _run_interface(self, runtime):
        import h5py
        import surfer.utils as surfutils
        import scipy.sparse
        import scipy.signal
        from cortex.polyutils import Surface

        in_ts = h5py.File(self.inputs.in_file, 'r')
        in_data = in_ts[self.inputs.data_field]
        out_file = self._list_outputs()['out_file']
        out_ts = h5py.File(out_file)

        in_ts.copy('COORDINATES',out_ts)
        in_ts.copy('STRUCTURES',out_ts) # TODO: fix reference copying fails
        
        structs = in_ts['STRUCTURES']

        fmri_group = out_ts.create_group('FMRI')
        nfeatures, nsamples = in_data.shape
        out_data = fmri_group.create_dataset(
            'DATA', dtype=np.float32,
            shape=(nfeatures,nsamples),
            maxshape=(nfeatures,None))
        
        for k,v in in_data.attrs.items():
            out_data.attrs[k] = v

        stdmap = np.std(in_data,-1)
        good_voxels = stdmap < self.inputs.std_threshold * stdmap[np.logical_not(np.isnan(stdmap))].mean()
        good_voxels[np.isnan(good_voxels)] = False

        for st in structs:
            attrs = structs[st].attrs
            if attrs['ModelType'] == 'SURFACE':
                sl = slice(attrs['IndexOffset'],
                           attrs['IndexOffset']+attrs['IndexCount'])
                # TODO, move to real heat kernel on surfaces

                """
                adj_mat = surfutils.mesh_edges(
                    np.asarray(structs[st]['TRIANGLES']))
                smooth_mat = surfutils.smoothing_matrix(
                    np.where(good_voxels[sl])[0],
                    adj_mat,
                    self.inputs.smoothing_steps)
                del adj_mat
                # TODO: see if it buffers in memory or not, if so iterate
                # over slabs of data (find optimal)

                sdata =  scipy.signal.detrend(
                    smooth_mat.dot(in_data[sl][good_voxels[sl]]),-1)
                del smooth_mat
                sdata -= sdata.mean(-1)[:,np.newaxis]
                sdata /= sdata.std(-1)[:,np.newaxis]
                sdata[np.isnan(sdata)] = 0
                """
                surf = Surface(np.asarray(in_ts['COORDINATES'][sl]),
                               np.asarray(structs[st]['TRIANGLES']))
                sdata = np.empty((attrs['IndexCount'],in_data.shape[1]))
                frame = np.empty(attrs['IndexCount'],dtype=in_data.dtype)
                for fr in xrange(in_data.shape[1]):
                    frame[:] = in_data[sl,fr]
                    frame[np.isnan(frame)]=0
                    sdata[:,fr] = surf.smooth(frame, self.inputs.smoothing_factor)
                del surf, frame
                #sdata[:] = scipy.signal.detrend(sdata,-1)
                #sdata -= sdata.mean(-1)[:,np.newaxis]
#                sdata /= sdata.std(-1)[:,np.newaxis]
                sdata[np.isnan(sdata)] = 0
                out_data[sl] = sdata 
                del sdata
            elif attrs['ModelType'] == 'VOXELS':
                # voxsize should be stored at sampling for convenience
                voxsize = 2.0 # could be anisotropic if necessary, see below
                for roi_name, label, ofst, cnt in structs[st]['ROIS']:
                    if cnt == 0:
                        continue
                    sl = slice(ofst, ofst+cnt)
                    coords = in_ts['COORDINATES'][sl]
                    adj_mat = scipy.sparse.coo_matrix(np.all(
                        np.abs(coords[np.newaxis] -
                               coords[:,np.newaxis])<voxsize*1.5, -1))
                    smooth_mat = surfutils.smoothing_matrix(
                        np.where(good_voxels[sl])[0],
                        adj_mat,
                        self.inputs.smoothing_steps)
                    del adj_mat
                    # TODO: see if it buffers in memory or not, if so iterate
                    # over slabs of data (find optimal)
                    #sdata =  scipy.signal.detrend(
                    #    smooth_mat.dot(in_data[sl][good_voxels[sl]]),-1)
                    sdata = smooth_mat.dot(in_data[sl][good_voxels[sl]])
                    #sdata -= sdata.mean(-1)[:,np.newaxis]
                    #sdata /= sdata.std(-1)[:,np.newaxis]
                    sdata[np.isnan(sdata)] = 0
                    out_data[sl] = sdata 
                    del smooth_mat
        if isdefined(self.inputs.TR):
            tr = self.inputs.TR
        elif 'TR' in in_ts['FMRI'].attrs.keys():
            tr = out_ts['FMRI'].attrs['TR']
        else:
            raise ValueError('TR is not known')
        
        from scipy import signal

        ub_frac = 1.
        if self.inputs.filter_range[1] is not -1:
            ub_frac = self.inputs.filter_range[1] * tr * 2.
        lb_frac = self.inputs.filter_range[0] * tr * 2.
        if lb_frac > 0 and ub_frac < 1:
            wp = [lb_frac, ub_frac]
            ws = [np.max([lb_frac - 0.1, 0]),np.min([ub_frac + 0.1, 1.0])]
        elif lb_frac == 0:
            wp = ub_frac
            ws = np.min([ub_frac + 0.1, 0.9])
        elif ub_frac == 1:
            wp = lb_frac
            ws = np.max([lb_frac - 0.1, 0.1])
        b, a = signal.iirdesign(wp, ws, 1, 60,ftype='ellip')
        print(b,a)

#        from scipy.fftpack import dct, idct
#        for i in xrange(out_ts['FMRI/DATA'].shape[0]):
#            tmp = signal.filtfilt(
#                b, a, np.asarray(out_ts['FMRI/DATA'][i]))
#            out_ts['FMRI/DATA'][i] = (tmp-tmp.mean())/tmp.std()

        """
        ts_dct = dct(out_ts['FMRI/DATA'], axis=1)
        cutoff = np.round(out_ts['FMRI/DATA'].shape[1] * tr / 128)
        ts_dct[:,:cutoff] = 0
        ts_dct = idct(ts_dct,axis=1)
        out_ts['FMRI/DATA'][:] = (ts_dct-ts_dct.mean(1)[:,np.newaxis])/ts_dct.std(1)[:,np.newaxis]
        out_ts['FMRI/DATA'][np.isnan(out_ts['FMRI/DATA'])] = 0 
        del ts_dct
        """
        in_ts.close()
        out_ts.close()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs["out_file"] = os.path.abspath('ts_smooth_bp.h5')
        return outputs



def fmri_surface_preproc(name='fmri_surface_preproc'):
    inputs = ['reference_boundary', 'init_reg','dicom_files',
              'fieldmap','fieldmap_reg','surfaces_volume_reference','mask',
              'resample_surfaces','resample_rois',
              'echo_spacing','repetition_time','echo_time',
              'phase_encoding_dir',
              'partial_volume_maps']

    inputnode = pe.Node(
        utility.IdentityInterface(
            fields=inputs),
        run_without_submitting = True,
        name='inputspec')
    outputnode = pe.Node(
        utility.IdentityInterface(
            fields=['timeseries','motion']),
        run_without_submitting = True,
        name='outputspec')

    n_motion_corr = pe.MapNode(
        nipy.preprocess.OnlinePreprocessing(
            out_file_format = 'ts.h5',
            resampled_first_frame = 'frame1.nii.gz',
            boundary_sampling_distance=2),
        overwrite=False,
        iterfield = ['dicom_files',
                     'fieldmap','fieldmap_reg','init_reg'],
        name = 'motion_correction')

    n_noise_corr = pe.MapNode(
        nipy.preprocess.OnlineFilter(
            out_file_format = 'ts.h5',
            resampled_first_frame = 'frame1.nii.gz',),
        iterfield = ['dicom_files',
                     'fieldmap','fieldmap_reg', 'motion'],
        overwrite=False,
        name = 'noise_correction')

    n_smooth_bp = pe.MapNode(
        GrayOrdinatesBandPassSmooth(
            data_field = 'FMRI/DATA_FILTERED',
            smoothing_factor=3,
#            smoothing_steps=3,
            filter_range=(.008,.1),
            TR=2.16),
        iterfield = ['in_file'],
        name = 'smooth_bp')


    n_smooth5_bp = pe.MapNode(
        GrayOrdinatesBandPassSmooth(
            data_field = 'FMRI/DATA_FILTERED',
            smoothing_factor=3,
#            smoothing_steps=5,
            filter_range=(.008,.1),
            TR=2.16),
        iterfield = ['in_file'],
        name = 'smooth5_bp')

    w=pe.Workflow(name=name)

    w.connect([
            (inputnode, n_motion_corr, [(x,x) for x in inputs[:-1]]),
            (inputnode, n_noise_corr, [(x,x) for x in inputs[2:]]),
            (n_motion_corr, n_noise_corr,[('motion',)*2]),
            (n_noise_corr, n_smooth_bp,[('out_file','in_file')]),
            (n_noise_corr, n_smooth5_bp,[('out_file','in_file')])
        ])
    return w


def onsets2regs(onsets,tr,nframes):
    import nipy.modalities.fmri.hemodynamic_models as hm
    hrf_regs = np.empty((onsets.size,nframes))
    for i,o in enumerate(onsets):
        hrf_regs[i]=hm.glover_hrf(
            tr, oversampling=1,
            time_length=(nframes)*tr, onset=o)
    return hrf_regs

#from sklearn import linear_model
#ridge = linear_model.Ridge(alpha = .5)
#clf.fit(hrf_regs.T,np.vstack((fts_lh,fts_rh)).T)


def fmri_goodvox(in_file, mask_file, mask_threshold=.1, factor=.5,neigh_smooth=2):
    from nipype.utils.filemanip import fname_presuffix
    import nibabel as nb
    import os
    import scipy.ndimage
    import numpy as np
    out_file = os.path.abspath(fname_presuffix(in_file,newpath='./',suffix='_goodvox'))
    nii = nb.load(in_file)
    data = nii.get_data()
    diffmap = np.abs(np.diff(data/data.mean(-1)[...,np.newaxis],1,-1)).mean(-1)
    
    mask = nb.load(mask_file).get_data() > mask_threshold
    diffmap_mask = diffmap * mask
    diffmap_norm = diffmap_mask/np.nanmean(diffmap[mask])
    diffmap_norm[np.isnan(diffmap_norm)]=0
    diffmap_sm = scipy.ndimage.filters.gaussian_filter(diffmap_norm,neigh_smooth)/\
                 scipy.ndimage.filters.gaussian_filter(mask.astype(np.float),neigh_smooth)
    diffmap_norm = diffmap / np.nanmean(diffmap[mask])/ diffmap_sm
    diffmap_norm *= mask
    
    std = np.nanstd(diffmap_norm[mask])
    mean = np.nanmean(diffmap_norm[mask])
    goodvox = (diffmap_norm < mean+std)*mask
    nb.Nifti1Image(goodvox.astype(np.uint8),nii.affine).to_filename(out_file)
    return out_file

def get_c_ras(mgz_file):
    import nibabel as nb
    import os
    import numpy as np
    mgz = nb.load(mgz_file)
    c_ras = np.eye(4)

    c_ras[:3,3] = mgz.header['Pxyz_c']
    out_fname = os.path.abspath('c_ras.txt')
    np.savetxt(out_fname ,c_ras, str('%f'))
    return out_fname

def wb_command_surface_apply_affine(in_file,c_ras):
    from subprocess import call
    import os
    from nipype.utils.filemanip import fname_presuffix
    out_file = os.path.abspath(fname_presuffix(in_file,newpath='./'))
    call(['wb_command','-surface-apply-affine', in_file, c_ras, out_file])
    return out_file

def wb_command_surface_resample(in_file,sphere_in,sphere_out,suffix):
    from subprocess import call
    import os
    from nipype.utils.filemanip import fname_presuffix
    out_file = os.path.abspath(fname_presuffix(in_file,newpath='./',suffix=suffix))
    call(['wb_command', '-surface-resample', in_file,sphere_in,sphere_out, 'BARYCENTRIC',out_file])
    return out_file

def wb_command_sphere_project_unproject(
    sphere_in,
    sphere_project_to,
    sphere_unproject_from,
    suffix):
    from subprocess import call
    import os
    from nipype.utils.filemanip import fname_presuffix
    sphere_out = os.path.abspath(fname_presuffix(sphere_in,newpath='./',suffix=suffix))
    call(['wb_command', '-surface-sphere-project-unproject', sphere_in, sphere_project_to, sphere_unproject_from,sphere_out])
    return sphere_out

def wb_command_label_resample(in_file,sphere_in,sphere_out,suffix):
    from subprocess import call
    import os
    from nipype.utils.filemanip import fname_presuffix
    out_file = os.path.abspath(fname_presuffix(in_file,newpath='./',suffix=suffix))
    call(['wb_command', '-label-resample', in_file, sphere_in, sphere_out, 'BARYCENTRIC', out_file, '-largest'])
    return out_file

def wb_command_volume_to_surface_mapping(
        in_file, surface, method,
        inner_surface=None, outer_surface=None,
        mask=None):
    from subprocess import call
    import os
    from nipype.utils.filemanip import fname_presuffix
    out_file = os.path.abspath(fname_presuffix(in_file, newpath='./', suffix='.gii', use_ext=False))
    cmd = ['wb_command', '-volume-to-surface-mapping', in_file, surface , out_file, '-%s'%method ]
    if method=='ribbon-constrained':
        cmd += [inner_surface, outer_surface]
        if not mask is None:
            cmd += ['-volume-roi', mask]
    call(cmd)
    return out_file
    
def wb_command_metric_smoothing(
        in_file,
        surface_file,
        smoothing_kernel=2,
        method='GEO_GAUSS_AREA'):
    from subprocess import call
    import os
    from nipype.utils.filemanip import fname_presuffix
    out_file = os.path.abspath(fname_presuffix(in_file, newpath='./', suffix='_smooth'))
    cmd = ['wb_command', '-metric-smoothing', surface_file, in_file , str(smoothing_kernel), out_file, '-method', method ]
    call(cmd)
    return out_file


def surface_32k(name='surface_32k', templates_dir='/home/bpinsard/data/src/Pipelines'):
    w=pe.Workflow(name=name)
    
    fs_templates = dict(
        white = [['subject','surf','?h.white']],
        pial = [['subject','surf','?h.pial']],
        sphere = [['subject','surf','?h.sphere.reg']],
        finalsurf = [['subject','mri','brain.finalsurfs.mgz']],
        aparc_a2009s_annot = [['subject','label','?h.aparc.a2009s.annot']],
        ba_annot = [['subject','label','?h.BA_exvivo.annot']],
        ba_thresh_annot = [['subject','label','?h.BA_exvivo.thresh.annot']]
    )

    n_fs_source = pe.Node(
        nio.DataGrabber(infields=['subject'],
                        outfields=fs_templates.keys(),
                        raise_on_empty=False,
                        sort_filelist=True, template='%s/%s/%s'),
        run_without_submitting=True,
        name='fs_source')
    n_fs_source.inputs.template_args = fs_templates

    n_c_ras = pe.Node(
        utility.Function(input_names=['mgz_file'],
                         output_names=['c_ras'],
                         function=get_c_ras),
        run_without_submitting=True,
        name='c_ras')
    
    n_white_to_gifti = pe.MapNode(
        freesurfer.MRIsConvert(out_datatype = 'gii'),
        iterfield = ['in_file'],
        name='white_to_gifti')
    n_pial_to_gifti = n_white_to_gifti.clone('pial_to_gifti')
    n_sphere_to_gifti = n_white_to_gifti.clone('sphere_to_gifti')
 
    n_sphere_project_unproject = pe.MapNode(
        utility.Function(
            input_names=[
                'sphere_in',
                'sphere_project_to',
                'sphere_unproject_from',
                'suffix'],
            output_names=['out_file'],
            function=wb_command_sphere_project_unproject),
        iterfield = ['sphere_in','sphere_project_to','sphere_unproject_from'],
        name='sphere_project_unproject')

    n_sphere_project_unproject.inputs.sphere_project_to = [os.path.join(templates_dir,'global/templates/standard_mesh_atlases/fs_%s/fsaverage.%s.sphere.164k_fs_%s.surf.gii'%(h,h,h)) for h in 'LR']
    n_sphere_project_unproject.inputs.sphere_unproject_from = [os.path.join(templates_dir,'global/templates/standard_mesh_atlases/fs_%s/fs_%s-to-fs_LR_fsaverage.%s_LR.spherical_std.164k_fs_%s.surf.gii'%(h,h,h,h)) for h in 'LR']
    n_sphere_project_unproject.inputs.suffix = '.proj_unproj'

    n_labels_to_gifti = pe.MapNode(
        freesurfer.MRIsConvert(out_datatype = 'gii'),
        iterfield=['annot_file','in_file'],
        name='labels_to_gifti')
    n_ba_to_gifti = n_labels_to_gifti.clone('ba_to_gifti')
    n_ba_thresh_to_gifti = n_labels_to_gifti.clone('ba_thresh_to_gifti')

    n_white_apply_affine = pe.MapNode(
        utility.Function(input_names=['in_file','c_ras'],
                         output_names=['out_file'],
                         function=wb_command_surface_apply_affine),
        iterfield = ['in_file'],
        name='white_apply_affine')
    
    n_pial_apply_affine = n_white_apply_affine.clone('pial_apply_affine')

    n_white_resample_surf = pe.MapNode(
         utility.Function(input_names=['in_file','sphere_in','sphere_out','suffix'],
                         output_names=['out_file'],
                         function=wb_command_surface_resample),
        iterfield = ['in_file','sphere_in','sphere_out'],
        name='white_resample_surf')
    n_white_resample_surf.inputs.suffix = '.32k'
    n_white_resample_surf.inputs.sphere_out = [os.path.join(templates_dir,'global/templates/standard_mesh_atlases/%s.sphere.32k_fs_LR.surf.gii'%h) for h in 'LR']
    n_pial_resample_surf = n_white_resample_surf.clone('pial_resample_surf')

    n_label_resample = pe.MapNode(
         utility.Function(input_names=['in_file','sphere_in','sphere_out','suffix'],
                         output_names=['out_file'],
                         function=wb_command_label_resample),
        iterfield = ['in_file','sphere_in','sphere_out'],        
        name='label_resample')
    n_label_resample.inputs.suffix = '.32k'
    n_label_resample.inputs.sphere_out = [os.path.join(templates_dir,'global/templates/standard_mesh_atlases/%s.sphere.32k_fs_LR.surf.gii'%h) for h in 'LR']

    n_ba_resample = n_label_resample.clone('BA_resample')
    n_ba_thresh_resample = n_label_resample.clone('BA_thresh_resample')

    w.connect([
            (n_fs_source,n_c_ras,[('finalsurf','mgz_file')]),

            (n_fs_source,n_white_to_gifti,[('white','in_file')]),
            (n_fs_source,n_labels_to_gifti,[('white','in_file'),
                                            ('aparc_a2009s_annot','annot_file')]),
            (n_fs_source,n_ba_to_gifti,[('white','in_file'),
                                        ('ba_annot','annot_file')]),
            (n_fs_source,n_ba_thresh_to_gifti,[('white','in_file'),
                                               ('ba_thresh_annot','annot_file')]),

            (n_fs_source,n_pial_to_gifti,[('pial','in_file')]),
            (n_fs_source,n_sphere_to_gifti,[('sphere','in_file')]),

            (n_sphere_to_gifti,n_sphere_project_unproject,[('converted','sphere_in')]),


            (n_white_to_gifti,n_white_apply_affine,[('converted','in_file')]),
            (n_pial_to_gifti,n_pial_apply_affine,[('converted','in_file')]),
            (n_c_ras, n_white_apply_affine,[('c_ras',)*2]),
            (n_c_ras, n_pial_apply_affine,[('c_ras',)*2]),

#           (n_white_to_gifti,n_white_resample_surf,[('converted','in_file')]),
#           (n_pial_to_gifti,n_pial_resample_surf,[('converted','in_file')]),
            (n_white_apply_affine,n_white_resample_surf,[('out_file','in_file')]),
            (n_pial_apply_affine,n_pial_resample_surf,[('out_file','in_file')]),

            (n_sphere_project_unproject,n_white_resample_surf,[('out_file','sphere_in')]),
            (n_sphere_project_unproject,n_pial_resample_surf,[('out_file','sphere_in')]),

            (n_sphere_project_unproject,n_label_resample,[('out_file','sphere_in')]),
            (n_labels_to_gifti,n_label_resample,[('converted','in_file')]),

            (n_sphere_project_unproject,n_ba_resample,[('out_file','sphere_in')]),
            (n_sphere_project_unproject,n_ba_thresh_resample,[('out_file','sphere_in')]),

            (n_ba_to_gifti,n_ba_resample,[('converted','in_file')]),
            (n_ba_thresh_to_gifti,n_ba_thresh_resample,[('converted','in_file')]),
            
            ])
    return w


# this function converts from itk world to nii world coords
# the atlas file has already been mapped from nii to itk world space 
# to be input of antsApplyTransformsToPoints
def coords_itk2nii(in_file):
    import numpy as np
    import os
    coords=np.loadtxt(in_file,skiprows=1,delimiter=',')
    coords[:,:2]=-coords[:,:2]
    out_file=os.path.abspath('atlas_coords_nii.csv')
    np.savetxt(out_file, coords, 
               header='x,y,z,r,s,t,l',
               fmt=bytes('%f,%f,%f,%d,%d,%d,%d'),
               delimiter=',')
    return out_file

def ants_for_subcortical(name='ants_for_subcortical'):

    input_node = pe.Node(
        utility.IdentityInterface(fields=['t1','template','coords']),
        name='inputspec')

    n_ants_2mni = pe.Node(
        ants.Registration(
            output_warped_image = 'INTERNAL_WARPED.nii.gz',
            output_transform_prefix = "anat2mni_",
            transforms = ['Translation', 'Rigid', 'Affine', 'SyN'],
            transform_parameters = [(0.1,), (0.1,), (0.1,), (0.2, 3.0, 0.0)],
            number_of_iterations = ([[10000, 111110, 11110]]*3 + [[100, 50, 30]]),
            dimension = 3,
            num_threads=8,
            write_composite_transform = True,
            collapse_output_transforms = False,
            metric = ['Mattes'] * 3 + [['Mattes', 'CC']],
            metric_weight = [1] * 3 + [[0.5, 0.5]],
            radius_or_number_of_bins = [32] * 3 + [[32, 4]],
            sampling_strategy = ['Regular'] * 3 + [[None, None]],
            sampling_percentage = [0.3] * 3 + [[None, None]],
            convergence_threshold = [1.e-8] * 3 + [-0.01],
            convergence_window_size = [20] * 3 + [5],
            smoothing_sigmas = [[4, 2, 1]] * 3 + [[1, 0.5, 0]],
            sigma_units = ['vox'] * 4,
            shrink_factors = [[6, 4, 2]] + [[3, 2, 1]]*2 + [[4, 2, 1]],
            use_estimate_learning_rate_once = [True] * 4,
            use_histogram_matching = [False] * 3 + [True],
            initial_moving_transform_com = True ),
        name='ants_2mni')

    n_warp_subctx = pe.Node(
        ants.ApplyTransformsToPoints(
            dimension=3,
            invert_transform_flags=[False]),
        name='warp_subctx')

    
    n_coords_itk2nii = pe.Node(
        utility.Function(
            input_names = ['in_file'],
            output_names = ['out_file'],
            function = coords_itk2nii),
        name='coords_itk2nii')

    w = pe.Workflow(name=name)

    w.connect([
            (input_node, n_ants_2mni,[('template','fixed_image'),
                                      ('t1','moving_image')]),
            (n_ants_2mni, n_warp_subctx,[(('composite_transform',wrap(filename_to_list),[]),'transforms')]),
            (input_node, n_warp_subctx,[('coords','input_file')]),
            (n_warp_subctx, n_coords_itk2nii, [('output_file','in_file')]),
            ])
    return w


def convert_motion_par(motion,epi2t1,matrix_file):
    from os.path import abspath
    import numpy as np 
    import nibabel as nb
    from nipy.algorithms.registration.affine import to_matrix44
    from scipy.signal import medfilt
    mot = np.loadtxt(motion)
    #mot = np.asarray([medfilt(p,3) for p in mot.T]).T
    epi2t1reg = np.loadtxt(epi2t1)
    nii = nb.load(matrix_file)
    reg = nii.affine
    nslices = nii.shape[2]
    mats = np.array([epi2t1reg.dot(np.linalg.inv(to_matrix44(m)).dot(reg)) for m in mot for i in range(nslices)])
    out_fname = abspath("motion.npy")
    np.save(out_fname,mats)
    return out_fname

n_convert_motion_par = pe.MapNode(
    utility.Function(
        input_names = ['motion','epi2t1','matrix_file'],
        output_names = ['motion'],
        function=convert_motion_par),
    iterfield=['motion','matrix_file','epi2t1'],
    name='convert_motion_par')



def workbench_pipeline(name='wb_pipeline', use_mask=False):

    input_node = pe.Node(
        utility.IdentityInterface(fields=[
            'lowres_rois_coords',
            'lowres_surf_lh',
            'lowres_surf_rh',
            'in_files',
            'masks']),
        name='inputspec')
    

    n_coords2fakesurf = pe.Node(
        utility.Function(
            input_names = ['in_file'],
            output_names = ['out_file'],
            function = coords2fakesurf),
        name='coords2fakesurf'
    )


    itfd = ['in_file']
    if use_mask:
        itfd.append('mask')
    n_volume2surface = pe.MapNode(
        utility.Function(
            input_names = ['in_file','surface','method','inner_surface','outer_surface','mask'],
            output_names = ['out_file'],
            function = wb_command_volume_to_surface_mapping
        ),
        iterfield=itfd,
        name='volume2surface'
    )
    n_volume2surface_lh = n_volume2surface.clone('volume2surface_lh')
    n_volume2surface_rh = n_volume2surface.clone('volume2surface_rh')
    n_volume2surface_lh.interface.inputs.method = n_volume2surface_rh.interface.inputs.method = 'ribbon-constrained'
    n_volume2surface_sc = n_volume2surface.clone('volume2surface_sc')
    n_volume2surface_sc.interface.inputs.method = 'trilinear'
    n_volume2surface_sc.iterfield = ['in_file']


    n_merge_gii = pe.Node(
        utility.Function(
            input_names = ['lh_tss','rh_tss','sc_tss'],
            output_names = ['grouped_tss'],
            function_str = 'def merge_gii(lh_tss,rh_tss,sc_tss): return [(l,r,s) for l,r,s in zip(lh_tss,rh_tss,sc_tss)]',
        ),
        name='merge_gii')


    w = pe.Workflow(name=name)

    w.connect([

        (input_node, n_coords2fakesurf,[('lowres_rois_coords','in_file')]),
        (input_node, n_volume2surface_lh,[
            (('lowres_surf_lh',utility.select,0),'surface'),
            (('lowres_surf_lh',utility.select,0),'inner_surface'),
            (('lowres_surf_lh',utility.select,1),'outer_surface'),
        ]),
        (input_node, n_volume2surface_rh,[
            (('lowres_surf_rh',utility.select,0),'surface'),
            (('lowres_surf_rh',utility.select,0),'inner_surface'),
            (('lowres_surf_rh',utility.select,1),'outer_surface'),
        ]),
        (n_coords2fakesurf, n_volume2surface_sc,[('out_file','surface')]),

        (n_volume2surface_lh, n_merge_gii,[('out_file','lh_tss')]),
        (n_volume2surface_rh, n_merge_gii,[('out_file','rh_tss')]),
        (n_volume2surface_sc, n_merge_gii,[('out_file','sc_tss')]),
        
    ])

    for n in [n_volume2surface_lh,n_volume2surface_rh,n_volume2surface_sc]:
        w.connect([(input_node,n,[('in_files','in_file')]),])
        if use_mask:
            w.connect([(input_node,n,[('masks','mask')]),])

    return w


def coords2fakesurf(in_file):
    import os
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix
    subcoords = np.loadtxt(in_file, delimiter=',', usecols=(0,1,2))
    points_da = nb.gifti.GiftiDataArray(subcoords[:,:3].astype(np.float32), 'pointset')
    points_da.ext_offset = ''
    tris_da = nb.gifti.GiftiDataArray(np.arange(6,dtype=np.int32).reshape(-1,3),'triangle')
    tris_da.ext_offset = ''
    fake_surf = nb.gifti.GiftiImage(darrays=[points_da, tris_da])
    out_fname = os.path.abspath(fname_presuffix(in_file, newpath='./', suffix='.gii', use_ext=False))
    nb.save(fake_surf, out_fname)
    return out_fname



def apply_affine(in_file,matrix):
    import os
    import numpy as np
    import nibabel as nb
    from nipype.utils.filemanip import fname_presuffix
    nii = nb.load(in_file)
    out_filename = os.path.abspath(fname_presuffix(in_file, newpath='./', suffix='_reg'))
    aff = np.loadtxt(matrix)
    nb.Nifti1Image(nii.get_data(), aff.dot(nii.affine)).to_filename(out_filename) 
    return out_filename

