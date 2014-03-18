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
            label_cnt += 1
            nlabels[labels==label] = label_cnt
            continue
        verts_mask = labels==label
        nverts = np.count_nonzero(verts_mask)
        if nverts == 0:
            continue
        center = np.mean(vertices[verts_mask], axis=0)
        centered_points = vertices[verts_mask] - center
        normal = center / scipy.linalg.norm(center)
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
        label_cnt += 1
        label_a.fill(label_cnt)
        for c in sorted(cuts):
            label_cnt += 1
            label_a[proj>c] = label_cnt
        nlabels[verts_mask] = label_a

    return nlabels
