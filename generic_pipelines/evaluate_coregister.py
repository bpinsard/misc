from nipype.interfaces import spm
import nipype.pipeline.engine as pe


def evaluate_coregister_wf(name='evaluate_coregister'):
    
    w=pe.Workflow(name=name)
    n_epi_grey_to_t1 = pe.Node(
        spm.Coregister(jobtype='write'),
        name='warp')
    w.add_nodes([n_epi_grey_to_t1])

    return w
