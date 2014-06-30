import warnings
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
    if partitions[0][0] > 0:
        label_cnt = 1

    verts_mask=np.empty(labels.shape,dtype=np.bool)
    for label, nparts in partitions:
        if nparts == 0:
            continue
        verts_mask[:] = labels==label
        if nparts == 1 and label!=0:
            nlabels[verts_mask] = label_cnt
            label_cnt += 1
            continue
        nverts = np.count_nonzero(verts_mask)
        if nverts == 0:
            continue
        center = np.mean(vertices[verts_mask], axis=0)
        centered_points = vertices[verts_mask] - center
        normal = center / np.linalg.norm(center)
        # project all vertex coordinates on the tangential plane for this point
        q,_ = scipy.linalg.qr(normal[:, np.newaxis],mode='full')
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
#        yield proj
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
import sys


def surfparc2vol(
    lh_surf_file,rh_surf_file,lh_parc_file,rh_parc_file,parc_file,
    out_fname,
    mask=None,
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
    if mask is None:
        wmgmvox = np.argwhere(parc_data>11000)
    else:
        wmgmvox = np.argwhere(mask)
    wmgmcoords = nb.affines.apply_affine(parc.get_affine(),
                                         wmgmvox).astype(np.float32)
    
    rois = np.zeros(parc.shape,dtype=np.int32)

    # set labels for volume rois 
    for i in rois_labels:
        rois[parc_data==i] = i

    # set labels for cortical surfaces searching for nearest vertex
    i=0
    for v,c in zip(wmgmvox,wmgmcoords):
        i+=1
        if i%1000==0:
            sys.stdout.write(('\r%02.2f'%(100*i/float(len(wmgmvox))))+'% done')
            sys.stdout.flush()
        rois[v[0],v[1],v[2]] = ctx_labels[
            np.argmin(((ctx_coords-c)**2).sum(1))]

    nb.save(nb.Nifti1Image(rois, parc.get_affine(), parc.header), out_fname)




import numpy as np
import nibabel.gifti
import nibabel as nb
import scipy.sparse
import scipy.sparse.linalg


"""
lh_surf=nb.gifti.read('./100307/MNINonLinear/fsaverage_LR32k/100307.L.midthickness.32k_fs_LR.surf.gii')
lh_parc=nb.gifti.read('./100307/MNINonLinear/fsaverage_LR32k/100307.L.aparc.a2009s.32k_fs_LR.label.gii')
lh_parcd=lh_parc.darrays[0].data
mask=lh_parcd==74
"""

def split_label_graph(verts, tris, labels, partitions, reord_subs = False):
    nlabels = np.zeros(labels.shape, dtype=labels.dtype)
    conn = scipy.sparse.coo_matrix((
            np.ones(3*tris.shape[0]),
            (np.hstack([tris[:,:2].T.ravel(),tris[:,1]]),
             np.hstack([tris[:,1:].T.ravel(),tris[:,2]]))))
    adj = (conn+conn.T>0).tocsr().astype(np.float32)

    label_cnt = 0
    
    verts_mask = np.empty(labels.shape,dtype=np.bool)
    stats = dict()
    for label, nparts in partitions:
        stats[label] = dict()
        if nparts == 0:
            continue
        verts_mask[:] = labels == label
        points = verts[verts_mask]
        nverts = np.count_nonzero(verts_mask)
        stats[label]['vertex_count'] = np.count_nonzero(verts_mask)
        if nverts == 0:
            warnings.warn('zero vertices in a label', RuntimeWarning)
            continue
        if nparts == 1:
            nlabels[verts_mask] = label_cnt
            if  label != 0:
                label_cnt += 1
            continue
        rois_avg_size = nverts / float(nparts)

        ######## projection for main orientation of vertices
        center = np.mean(points, axis=0)
        centered_points = points - center
        normal = center / np.linalg.norm(center)
        q,_ = scipy.linalg.qr(normal[:, np.newaxis], mode='full')
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
        proj = np.dot(points, axis)

        ######### connect single vertices to closest vertex
        roi_graph = adj[verts_mask][:,verts_mask]
        unconn = np.where(np.asarray(roi_graph.sum(0)==0))[1]
        stats[label]['single_vertex'] = len(unconn)

        if len(unconn) > 0:
            print '%d unconnected vertices'%len(unconn)
            unconn_coords = points[unconn]
            nearest = np.argsort(
                np.linalg.norm(
                    unconn_coords[:,np.newaxis]-points[np.newaxis],
                    axis=-1),axis=-1)[:,1]
            print nearest
            roi_graph[unconn,nearest] = 1
            roi_graph[nearest,unconn] = 1

        ########## connect small connected components to closest conn. comp.
        while True:
            lap = scipy.sparse.csgraph.laplacian(roi_graph)
            neig = 5
            elap = np.linalg.svd(np.asarray(lap.todense()))
            null_thr = 1e-6
            nullspace = elap[0][:,elap[1]<null_thr]
            idx = np.squeeze(np.lexsort(nullspace.T))
            comps = np.empty(nverts, dtype=np.int)

            comps[idx] = np.cumsum(
                np.abs(np.ediff1d(nullspace.sum(1)[idx], to_begin=[0]))>1e-10)
            compcnts = np.bincount(comps)
            print compcnts
            if not stats[label].has_key('compcnts'):
                stats[label]['components_counts'] = compcnts

            small_comps = compcnts<rois_avg_size/2
            if np.count_nonzero(small_comps)>0:
                print 'small comps', compcnts
                smallest = np.argmin(compcnts)
                dists = np.linalg.norm(
                    points[comps==smallest, np.newaxis]-\
                        points[np.newaxis, comps!=smallest],
                    axis=-1)
                nearest = np.unravel_index(np.argmin(dists), dists.shape)
                r = np.where(comps==smallest)[0][nearest[0]]
                c = np.where(comps!=smallest)[0][nearest[1]]
                print 'connect %d %d : %f'%(r,c,dists[nearest])
                roi_graph[r,c] = 1
                roi_graph[c,r] = 1
            else:
                break

        print 'comps size', compcnts
        ncomps = np.max(comps)+1
        print '%d connected components in roi %d : %d parts'%(ncomps,label,nparts)
        if ncomps > nparts :
            raise RuntimeError('label %d:there are %d connected components for %d partitions only'%(label,ncomps,nparts))
        elif ncomps < nparts:
            comps_size = np.bincount(comps)
            toobigcomps = comps_size > rois_avg_size
            nsmallenough = (toobigcomps==0).sum()
            sz = comps_size[toobigcomps].sum()/float(nparts-nsmallenough)
            divs_float = np.ones(ncomps)
            divs_float[toobigcomps] = (comps_size[toobigcomps]/sz)
            divs_int = np.ceil(divs_float).astype(np.int)
            xceedparts = divs_int.sum() - nparts
            if xceedparts > 0:
                divs_int[np.argsort(
                        divs_float-divs_int+(toobigcomps==0))[:xceedparts]]-=1
            subs = np.zeros(nverts, dtype=np.int)
            nsub = 0
            print divs_int
            for i, divs in enumerate(divs_int):
                subverts_mask = comps==i
                if divs == 1:
                    subs[subverts_mask] = nsub
                    nsub += 1
                    continue
                subroi_graph = roi_graph[subverts_mask][:,subverts_mask]
                lap = scipy.sparse.csgraph.laplacian(subroi_graph)
                elap = np.linalg.svd(np.asarray(lap.todense()))
                fiedler = elap[0][:,-2]
                if proj[subverts_mask].dot(fiedler) < 0:
                    fiedler = -fiedler
                del elap
                thresh_idx = np.round(
                    comps_size[i]/divs*np.arange(divs)).astype(np.int)
                thresh = np.sort(fiedler)[thresh_idx]
                sub_mask = subverts_mask.copy()
                for t in thresh:
                    sub_mask[subverts_mask] = fiedler>=t
                    subs[sub_mask] = nsub
                    nsub += 1
        else:
            subs = comps
            nsub = ncomps

        #reorder for approximate correspondency between subjects ??!?
        if reord_subs:
            reord = np.argsort([np.median(proj[subs==i]) for i in range(nsub)])
            print 'reord', [proj[subs==i].mean() for i in range(nsub)], reord
            subs = reord[subs]            

        subs += label_cnt
        label_cnt += nsub
        nlabels[verts_mask] = subs
    return nlabels, stats
