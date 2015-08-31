import numpy as np
import os
import glob
import nibabel as nb
import nibabel.gifti

from traits.api import HasTraits, Instance, Array, Int, Float, \
    Bool, Dict, on_trait_change, Range, Property, Button
from traitsui.api import View, Item, HGroup, Group

from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MlabSceneModel

subjects_dir = '/home/bpinsard/softs/freesurfer/subjects'
subject = 'fs32k_new'

DEFAULT_LUT_MODE = 'RdBu'
DEFAULT_LUT_REVERSE = True

class HCPViewer():

    def __init__(self):

        

        self._scalar_range = np.array([0,1])

        lh_surf = nb.gifti.read(os.path.join(subjects_dir,subject,'surf','lh.white.smoothed.32k.gii'))
        rh_surf = nb.gifti.read(os.path.join(subjects_dir,subject,'surf','rh.white.smoothed.32k.gii'))
        
        shift = np.asarray([100,0,0])
        coords = np.vstack([
                lh_surf.darrays[0].data-shift,
                rh_surf.darrays[0].data+shift
                ])
        triangles = np.vstack([
                lh_surf.darrays[1].data,
                lh_surf.darrays[1].data+lh_surf.darrays[0].data.shape[0]])
        self._surf = mlab.triangular_mesh(coords[:,0], coords[:,1], coords[:,2], triangles)

        self._surf.module_manager.scalar_lut_manager.lut_mode = DEFAULT_LUT_MODE
        self._surf.module_manager.scalar_lut_manager.reverse_lut = DEFAULT_LUT_REVERSE
        self._surf.module_manager.scalar_lut_manager.use_default_range = False

        del coords, triangles
        
        rois_aparc = np.loadtxt(
            os.path.join(subjects_dir,subject,'label','Atlas_ROIs_new.csv'), 
            skiprows=1, delimiter=',')
#        rois_aparc = rois_aparc[rois_aparc[:,-1]==8]
        cen = np.vstack([rois_aparc[:,:3],lh_surf.darrays[0].data,rh_surf.darrays[0].data]).mean(0)
        uniqlabels = np.unique(rois_aparc[:,-1])
        coords = rois_aparc[:,:3].copy()
        cens = np.asarray([coords[rois_aparc[:,-1]==l].mean(0) for l in uniqlabels])

        for l,c in zip(uniqlabels, cens):
            coords[rois_aparc[:,-1]==l] += c-cen

        self._pts = mlab.points3d(
            coords[:,0], coords[:,1], coords[:,2],
            mode='cube', scale_factor=1)

        self._pts.scene.background = (0.0, 0.0, 0.0)
        self._pts.module_manager.scalar_lut_manager.lut_mode = DEFAULT_LUT_MODE
        self._pts.module_manager.scalar_lut_manager.reverse_lut = DEFAULT_LUT_REVERSE
        self._pts.module_manager.scalar_lut_manager.use_default_range = False

        lut_path = os.path.join(os.environ['FREESURFER_HOME'], 'FreeSurferColorLUT.txt')
        lut_file = open(lut_path)
        self._lut = dict()
        for l in lut_file.readlines():
            if len(l)>4 and l[0]!='#':
                l = l.split()
                self._lut[int(l[0])] = (l[1],tuple(float(c)/255. for c in l[2:5]))
        lut_file.close()

        giis=[nibabel.gifti.read(glob.glob('/home/bpinsard/data/tests/label2surf/giis/%d_*ras.gii'%l)[0]) for l in uniqlabels]

        self._pts.scene.disable_render = True
        self._rois_surfaces = []
#        tr = np.asarray([[-1,0,0],[0,0,1],[0,-1,0]])
        for l,g,cc in zip(uniqlabels, giis,cens):
            coords=g.darrays[0].data+(cc-cen)#.dot(np.array([[-1,0,0],[0,0,1],[0,-1,0]]))##*(-1,1,1)
#            coords=np.array([[-1,0,0],[0,0,1],[0,1,0]]).dot(coords)
            surf = mlab.triangular_mesh(coords[:,0], coords[:,1], coords[:,2], g.darrays[1].data,opacity=.3)

            surf.actor.property.color = self._lut[l][1]
            surf.actor.mapper.scalar_visibility = False
            surf.actor.actor.position = [-3,3,3]
            surf.name = self._lut[l][0]
            self._rois_surfaces.append(surf)

        """
        for l,c in zip(uniqlabels, cens):
            mask = (hr_mask == l)
            aw = np.argwhere(mask)
            awmin,awmax = aw.min(0),aw.max(0)
            bbox = [slice(b,t) for b,t in zip(awmin,awmax)]
            del aw
            src = mlab.pipeline.scalar_field(mask[bbox].astype(np.uint8))
            src.name = 'rois_%d'%l
            surf = mlab.pipeline.iso_surface(
                src, contours = [1], opacity=0.3)
            surf.actor.property.color = self._lut[l][1]
            surf.actor.mapper.scalar_visibility = False

            surf.actor.actor.position = nb.affines.apply_affine(rois_mask_highres.get_affine(), awmin) + (cen - c)*[1,-1,-1]

            surf.actor.actor.scale = surf_spacing
            self._rois_surfaces.append(surf)
            del mask
            """
        self._pts.scene.disable_render = False


    def set_range(self, scalar_range):
        self._scalar_range = np.asarray(scalar_range)
        self._pts.scene.disable_render = True
        self._pts.module_manager.scalar_lut_manager.data_range = self._scalar_range
        self._surf.module_manager.scalar_lut_manager.data_range = self._scalar_range
        self._pts.glyph.glyph.range = self._scalar_range
        self._pts.glyph.glyph.scale_factor = 1./self._scalar_range[1]
        #self._pts.glyph.glyph.clamping = True
        self._pts.scene.disable_render = False

        
    def set_data(self, data):
        self._data = np.atleast_2d(data)
        self._data_idx = 0
        self._update(self._data[0])
        
    def set_data_idx(self, i):
        if i>=0 and i< len(self._data):
            self._data_idx = i
            self._update(self._data[i])
        else:
            raise ValueError
        
    def _update(self, data):
        self._pts.scene.disable_render = True
        self._surf.mlab_source.scalars = data[:2*32492]
        self._pts.mlab_source.scalars = data[2*32492:]
        self.set_range(self._scalar_range)
        self._pts.scene.disable_render = False
                
        
