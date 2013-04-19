from nipype.interfaces import spm, fsl, afni, nitime, utility, lif

import nipype.pipeline.engine as pe
import nipype.pipeline.file_proxy as fileproxy
import nipype.interfaces.io as nio
import nipype.workflows.rsfmri.spm as spmrest
import nipype.interfaces.nipy.utils as nipyutils
import nipype.interfaces.nipy.preprocess as nipypp
import nipype.utils.filemanip as fmanip
from .utils import *
import numpy as np

def regress_out_confounds(name='regout_confounds'):
    wkfl = pe.Workflow(name=name)
    inputnode = pe.Node(utility.IdentityInterface(
            fields=['in_file','mask','motion','csf_seg','wm_seg']),
                        name='inputspec')
    outputnode = pe.Node(utility.IdentityInterface(fields=['corrected_file']),
                         name='outputspec')

    def combine_regressors(motion,motion_source,regressors):
        import nipy.algorithms.utils.preprocess as preproc
        motion = preproc.motion_parameter_standardize(motion,motion_source)
        
    n_combine_regressors = pe.Node(
        utility.Function(input_names=['motion',
                                      'motion_source',
                                      'regressors'],
                         output_names=['regressors'],
                         function=combine_regressors))
    n_remove_confound = pe.Node(fsl.FilterRegressor(filter_all=True),
                              name='remove_confound',)
    
    

def compcorr(name='compcorr'):
    from nipype.workflows.rsfmri.fsl.resting import extract_noise_components
    from nipype.algorithms.misc import TSNR

    wkfl = pe.Workflow(name=name)
    inputnode = pe.Node(utility.IdentityInterface(fields=['in_file',
                                                       'mask',
                                                       'num_components']),
                        name='inputspec')
    outputnode = pe.Node(utility.IdentityInterface(fields=['corrected_file']),
                         name='outputspec')
                            
    tsnr = pe.Node(TSNR(), name='tsnr')
    getthresh = pe.Node(interface=fsl.ImageStats(op_string='-k %s -p 98'),
                        name='getthreshold')
    threshold_stddev = pe.Node(fsl.Threshold(), name='threshold')
    compcor = pe.Node(utility.Function(input_names=['realigned_file',
                                                    'noise_mask_file',
                                                    'num_components'],
                                       output_names=['noise_components'],
                                       function=extract_noise_components),
                      name='compcorr',)
    remove_noise = pe.Node(fsl.FilterRegressor(filter_all=True),
                           name='remove_noise',)

    wkfl.connect([
        (inputnode,tsnr,[('in_file','in_file')]),
        (inputnode, compcor, [('in_file','realigned_file'),
                              ('num_components','num_components')]),
        (tsnr, threshold_stddev,[('stddev_file', 'in_file')]),
        (tsnr, getthresh, [('stddev_file', 'in_file')]),
        (inputnode, getthresh, [('mask','mask_file')]),
        (inputnode, remove_noise, [('in_file','in_file')]),
        (getthresh, threshold_stddev,[('out_stat', 'thresh')]),
        (threshold_stddev, compcor, [('out_file',  'noise_mask_file')]),
        (compcor,remove_noise, [('noise_components', 'design_file')]),
        (inputnode, remove_noise, [('mask','mask')]),
        (remove_noise, outputnode, [('out_file','corrected_file')]),
        ])
    return wkfl

def corsica(name='corsica'):
    inputnode = pe.Node(utility.IdentityInterface(
            fields=['in_file','mask','noise_rois','mni_to_t1','tr','coregistered_fmri']),
                        name='inputspec')
    outputnode = pe.Node(utility.IdentityInterface(
            fields=['corrected_file','warped_noise_rois',
                    'sica_file','ica_components']),
                         name='outputspec')

    n_warp_noise_rois = pe.Node(
        spm.preprocess.ApplyDeformations(
            interp=0,),
        name='warp_noise_rois')

    n_ica=fileproxy.GunzipNode(
        interface=lif.SICA(),
        name='ica',)
    n_corsica=fileproxy.GunzipNode(
        interface=lif.CORSICA(score_thresh=0.25,add_residuals=True),
        name='corsica')

    w=pe.Workflow(name=name)
    w.connect([
       (inputnode,n_warp_noise_rois,[('mni_to_t1','deformation_field'),
                                     ('coregistered_fmri','reference_volume'),
                                     ('noise_rois','in_files')]),
       (inputnode,n_ica,[('in_file','in_file'), ('mask','mask'),('tr','TR')]),
       (inputnode, n_ica, [(('in_file',fname_presuffix_basename,'','_ics','.'),
                            'sica_comp_filename')]),
       (inputnode,n_corsica,[('in_file', 'in_file'),
                             (('in_file',fname_presuffix_basename,'c','','.'),
                              'corrected_file')]),
       (n_ica,n_corsica,[('sica_file', 'sica_file')]),
       (n_warp_noise_rois,n_corsica,[('out_files','noise_rois')]),
       (n_ica, outputnode, [('components','ica_components'),
                            ('sica_file','sica_file')]),
       (n_warp_noise_rois,outputnode,[('out_files','warped_noise_rois')]),
       (n_corsica, outputnode, [('corrected_file','corrected_file')])
       ])
    return w

def corsica2(name='corsica'):
    inputnode = pe.Node(utility.IdentityInterface(
            fields=['in_file','mask','noise_rois','t1_to_mni','t1_to_epi','tr','coregistered_fmri','t1_to_epi','t1']),
                        name='inputspec')
    outputnode = pe.Node(utility.IdentityInterface(
            fields=['corrected_file','warped_noise_rois',
                    'sica_file','ica_components']),
                         name='outputspec')

    n_warp_noise_rois = fileproxy.GunzipNode(
        spm.utils.ApplyInverseDeformation(
            interpolation=0,),
        name='warp_noise_rois')

    n_t1_to_fmri = pe.Node(
        fsl.FLIRT(interp='nearestneighbour',
                  apply_xfm=True,),
        name='t1_to_fmri')

    n_ica=fileproxy.GunzipNode(
        interface=lif.SICA(),
        name='ica',)
    n_corsica=fileproxy.GunzipNode(
        interface=lif.CORSICA(score_thresh=0.25,add_residuals=True),
        name='corsica')

    w=pe.Workflow(name=name)
    w.connect([
       (inputnode,n_warp_noise_rois,[('noise_rois','in_files'),
                                    ('t1_to_mni','deformation_field'),
                                    ('t1','target')]),
       (inputnode,n_t1_to_fmri,[('coregistered_fmri','reference'),
                                ('t1_to_epi','in_matrix_file')]),
       (n_warp_noise_rois, n_t1_to_fmri, [('out_files','in_file')]),
       
       (inputnode,n_ica,[('in_file','in_file'), ('mask','mask'),('tr','TR')]),
       (inputnode, n_ica, [(('in_file',fname_presuffix_basename,'','_ics','.'),
                            'sica_comp_filename')]),
       (inputnode,n_corsica,[('in_file', 'in_file'),
                             (('in_file',fname_presuffix_basename,'c','','.'),
                              'corrected_file')]),
       (n_ica,n_corsica,[('sica_file', 'sica_file')]),
       (n_t1_to_fmri,n_corsica,[('out_file','noise_rois')]),
       (n_ica, outputnode, [('components','ica_components'),
                            ('sica_file','sica_file')]),
       (n_warp_noise_rois,outputnode,[('out_files','warped_noise_rois')]),
       (n_corsica, outputnode, [('corrected_file','corrected_file')])
       ])
    return w



def connectivity_analysis(name):

    inputnode = pe.Node(utility.IdentityInterface(
            fields=['in_file','mask','rois_files','rois_labels','tr',
                    'networks']),
                        name='inputspec')
    outputnode = pe.Node(utility.IdentityInterface(
            fields=['correlations']),
                         name='outputspec')
    n_extract_ts = pe.Node(
        nitime.GetTimeSeries(
            aggregating_function = nitime.analysis.mean,),
        name='extract_ts')

    n_correlation_analysis = pe.Node(
        nitime.CorrelationAnalysis(            
            bootstrap_estimation = True,
            bootstrap_nsamples = 1000),
        name = 'correlation_analysis')
    n_integration_analysis = pe.Node(
        nitime.IntegrationAnalysis(),
        name='integration_analysis')

    w=pe.Workflow(name=name)
    w.connect([
        (inputnode,n_extract_ts,[('rois_files','rois_files'),
                                 ('rois_labels','rois_labels_files'),
                                 ('in_file','in_file'),
                                 ('tr','sampling_interval')]),
        (n_extract_ts, n_correlation_analysis, [('timeseries','timeseries')]),
        (n_correlation_analysis,outputnode,[('correlations','correlations')]),
#        (inputnode, n_integration_analysis,[('networks','networks')]),
#        (n_correlation_analysis,n_integration_analysis,
#         [('correlations','correlations_file')]),
        ])
    return w


def zscorr_analysis(name):

    inputnode = pe.Node(utility.IdentityInterface(
            fields=['in_file','seeds','distances','correlations','mask',
                    'rois_files','rois_labels']),
                        name='inputspec')
    outputnode = pe.Node(utility.IdentityInterface(
            fields=['zscore_file']),
                         name='outputspec')

    n_corrdist = pe.Node(
        nipyutils.CorrelationDistributionMaps(
            min_distance=30,
            out_dtype=np.float16,
            nbins=1000),
        name = 'corrdist_maps')

    def corr_to_zscore(in_file,distrib):
        import os
        import nipype.utils.filemanip as fmanip
        from nipype.utils.filemanip import fname_presuffix
        corrs=fmanip.loadpkl(in_file)
        dist =fmanip.loadpkl(distrib)
        zscore = (corrs['corr']-dist[0]['mean'])/dist[0]['std']
        zfile = os.path.abspath(fname_presuffix(in_file, suffix='_zscore',
                                                newpath=os.getcwd()))
        corrs['zscore']=zscore
        fmanip.savepkl(zfile,corrs)
        return zfile

    n_zscorr = pe.Node(
        utility.Function(input_names=['in_file','distrib'],
                         output_names=['zscore_file'],
                         function=corr_to_zscore),
        name = 'zscorr')

    w=pe.Workflow(name=name)
    w.connect([
        (inputnode, n_corrdist, [('in_file','in_files'),
                                 ('seeds','seed_masks'),
                                 ('distances','distances')]),
        (inputnode, n_corrdist, [('mask', 'mask')]),

        (inputnode,n_zscorr,[('correlations','in_file')]),
        (n_corrdist,n_zscorr,[(('global_correlation_distribution',
                                wrap(operator.getitem),[0]),'distrib')]),
        (n_zscorr,outputnode, [('zscore_file','zscore_file')])
        ])
    return w

def sample_seeds(name='sample_seeds'):
    w=pe.Workflow(name=name)
    n_sample_seeds = pe.Node(
        nipyutils.SampleSeedsMap(nsamples=1000),
        name='sample_seeds')

    w.add_nodes([n_sample_seeds])
    return w


def distrib_corr_conn(in_file,mask_file,rois_file,rois_labels,distrib):
    import nibabel as nb
    nii=nb.load(in_file)
    mask=nb.load(mask_file).get_data()>0
    data=nii.get_data()[mask]
    ndata=(data-data.mean(1))/data.std(1)
    rois=nb.load(rois_file).get_data()[mask]
    rois_ids = np.unique(rois)
