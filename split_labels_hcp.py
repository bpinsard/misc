import numpy as np
import scipy.linalg


"""
import nibabel as nb
import nibabel.gifti

subject = '100307'
path = '%s/MNINonLinear/fsaverage_LR32k/%s'%(subject,subject)
lh_parc=nb.gifti.read('%s.L.aparc.a2009s.32k_fs_LR.label.gii'%path)
lh_sphere=nb.gifti.read('%s.L.sphere.32k_fs_LR.surf.gii'%path)
rois_size_max=128
bins = np.bincount(lh_parc.darrays[0].data)[1:] # remove 0-labels
nparts = np.ceil(bins/float(rois_size_max)).astype(np.int)
divs = np.c_[np.arange(1,lh_parc.darrays[0].data.max()+1), nparts]
lh_nlabels = split_labels_hcp.split_label(
   lh_parc.darrays[0].data,
   lh_sphere.darrays[0].data,
   lh_sphere.darrays[1].data,divs)
"""

def split_label(labels, vertices, triangles, partitions):
    nlabels = np.zeros(labels.shape, dtype=labels.dtype)

    label_cnt = 0 

    for label, nparts in partitions:
        if nparts == 0:
            continue
        if nparts == 1:
            nlabels[labels==label] = label_cnt
            label_cnt += 1
            continue
        verts_mask = labels==label
        nverts = np.count_nonzero(verts_mask)
        if nverts == 0:
            continue
        center = np.mean(vertices[verts_mask], axis=0)
        centered_points = vertices[verts_mask] - center
        normal = center / np.linalg.norm(center)
        # project all vertex coordinates on the tangential plane for this point
        q,_ = scipy.linalg.qr(normal[:, np.newaxis])
        tangent_u = q[:, 1:]
        m_obs = np.dot(centered_points, tangent_u)
        # find principal eigendirection
        m_cov = np.dot(m_obs.T, m_obs)
        w, vr = scipy.linalg.eig(m_cov)
        i = np.argmax(w)
        eigendir = vr[:, i]
        # project back into 3d space
        axis = np.dot(tangent_u, eigendir)
        # orient them from posterior to anterior
        if axis[1] < 0:
            axis *= -1

        # project the label on the axis
        proj = np.dot(vertices[verts_mask], axis)
        cuts = np.percentile(proj,
                             (100*np.arange(1,nparts)/float(nparts)).tolist())
        label_a = np.empty(nverts, dtype=labels.dtype)
        label_a.fill(label_cnt)
        for c in sorted(cuts):
            label_cnt += 1
            label_a[proj>c] = label_cnt
        label_cnt += 1
        nlabels[verts_mask] = label_a

    return nlabels



import nibabel as nb
import nibabel.gifti
import numpy as np

def surfparc2vol(
    lh_surf_file,rh_surf_file,lh_parc_file,rh_parc_file,parc_file,
    out_fname,
    rois_labels = [8, 10, 11, 12, 13, 16, 17, 18, # HCP rois labels
                   26, 28, 47, 49, 50, 51, 52, 53, 54, 58, 60]):
    
    lh_parc = nb.gifti.read(lh_parc_file)
    rh_parc = nb.gifti.read(rh_parc_file)
    lh_surf = nb.gifti.read(lh_surf_file)
    rh_surf = nb.gifti.read(rh_surf_file)

    ctx_mask = np.hstack([lh_parc.darrays[0].data>0,
                          rh_parc.darrays[0].data>0])
    ctx_coords = np.vstack([lh_surf.darrays[0].data,
                            rh_surf.darrays[0].data])[ctx_mask]
    ctx_labels = np.hstack([lh_parc.darrays[0].data+11000,
                            rh_parc.darrays[0].data+12000])[ctx_mask]

    parc = nb.load(parc_file)
    parc_data = parc.get_data()
    wmgmvox = np.argwhere(parc_data>11000)
    wmgmcoords = nb.affines.apply_affine(parc.get_affine(),
                                         wmgmvox).astype(np.float32)
    
    rois = np.zeros(parc.shape,dtype=np.int32)

    # set labels for volume rois 
    for i in rois_labels:
        rois[parc_data==i] = i

    # set labels for cortical surfaces searching for nearest vertex
    for v,c in zip(wmgmvox,wmgmcoords):
        rois[v[0],v[1],v[2]] = ctx_labels[
            np.argmin(((ctx_coords-c)**2).sum(1))]

    nb.save(nb.Nifti1Image(rois, parc.get_affine()), out_fname)

