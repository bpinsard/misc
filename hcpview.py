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

from matplotlib.cm import get_cmap
from matplotlib import pyplot

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
subjects_dir = os.path.join(SRC_DIR, 'hcp_view_rois_surfs')
subject = 'fs32k'
DEFAULT_SURF = 'inflated.32k'
DEFAULT_CURV = 'curv.32k'

DEFAULT_BG_LUT_MODE = 'Greys'
DEFAULT_LUT_MODE = 'RdBu'
DEFAULT_LUT_NAME = 'plasma'
DEFAULT_LUT = get_cmap(DEFAULT_LUT_NAME)
DEFAULT_LUT_TABLE = DEFAULT_LUT(np.linspace(.3, 1., 256))*255
DEFAULT_LUT_TABLE = np.vstack([DEFAULT_LUT_TABLE[:128],np.zeros((1,4)),DEFAULT_LUT_TABLE[128:]])

DEFAULT_LUT_TABLE[0,-1] = 0 # transparent
DEFAULT_LUT_REVERSE = True
SUBCTX_OPACITY = .3

custom_shifts = {
    'Left-Cerebellum-Cortex': [-10,-78,0],
    'Right-Cerebellum-Cortex': [10,-78,0],
    'Left-Thalamus-Proper': [-15,-65,0],
    'Right-Thalamus-Proper': [15,-65,0],
    'Left-Caudate' : [-12, -4, 0],
    'Right-Caudate' : [12, -4, 0],
    'Left-Putamen' : [-28, -12, 0],
    'Right-Putamen' : [28, -12, 0],
    'Left-Pallidum' : [-23, -16, 0],
    'Right-Pallidum' : [23, -16, 0],
    'Left-Accumbens-area' : [-3, 8, 0],
    'Right-Accumbens-area' : [3, 8, 0],
    'Brain-Stem' : [0,-41,0],
    'Left-Hippocampus' : [-30,-50,0],
    'Right-Hippocampus' : [30,-50,0],
    'Left-Amygdala' : [-30,-42,0],
    'Right-Amygdala' : [30,-42,0],
    'Left-VentralDC' : [-10,-31,0],
    'Right-VentralDC' : [10,-31,0],
}


class HCPViewer():

    def __init__(self,
                 surf=DEFAULT_SURF,
                 lut_mode=DEFAULT_LUT_MODE, lut_reverse=DEFAULT_LUT_REVERSE,
                 bg_lut_mode=DEFAULT_BG_LUT_MODE):

        

        mlab.figure(size=(1000,1000))
        self._scalar_range = np.array([0,1])

        lh_surf = nb.load(os.path.join(subjects_dir,subject,'surf','lh.%s.gii'%surf))
        rh_surf = nb.load(os.path.join(subjects_dir,subject,'surf','rh.%s.gii'%surf))
        
        lh_curv = nb.load(os.path.join(subjects_dir,subject,'surf','lh.%s.gii'%DEFAULT_CURV))
        rh_curv = nb.load(os.path.join(subjects_dir,subject,'surf','rh.%s.gii'%DEFAULT_CURV))
        self._curv = np.hstack([lh_curv.darrays[0].data,rh_curv.darrays[0].data])
        
        shift = np.asarray([110,0,0])
        lh_coords = lh_surf.darrays[0].data#-shift
        rh_coords = rh_surf.darrays[0].data#+shift
        lh_tris = lh_surf.darrays[1].data
        rh_tris = rh_surf.darrays[1].data

        self._lh_bg_surf = mlab.triangular_mesh(lh_coords[:,0], lh_coords[:,1], lh_coords[:,2], lh_tris)
        self._rh_bg_surf = mlab.triangular_mesh(rh_coords[:,0], rh_coords[:,1], rh_coords[:,2], rh_tris)
        self._lh_bg_surf.mlab_source.scalars = lh_curv.darrays[0].data
        self._rh_bg_surf.mlab_source.scalars = rh_curv.darrays[0].data
        self._lh_bg_surf.module_manager.scalar_lut_manager.lut_mode = bg_lut_mode
        self._rh_bg_surf.module_manager.scalar_lut_manager.lut_mode = bg_lut_mode
        
        self._lh_surf = mlab.triangular_mesh(lh_coords[:,0], lh_coords[:,1], lh_coords[:,2], lh_tris)
        self._rh_surf = mlab.triangular_mesh(rh_coords[:,0], rh_coords[:,1], rh_coords[:,2], rh_tris)
        for surf in [self._lh_surf, self._rh_surf]:
            surf.module_manager.scalar_lut_manager.lut.table = DEFAULT_LUT_TABLE
            surf.module_manager.scalar_lut_manager.use_default_range = False
            surf.module_manager.scalar_lut_manager.lut.nan_color = [ 0.5,  0.5,  0.5,  0]
        self._scene = self._lh_surf.scene
        self._scene.disable_render = True
                
        for surf in [self._lh_surf, self._lh_bg_surf]:
            surf.actor.actor.position = [-120,-50,0]
        for surf in [self._rh_surf, self._rh_bg_surf]:
            surf.actor.actor.position = [120,-50,0]
            
        rois_aparc = np.loadtxt(
            os.path.join(SRC_DIR,'data','Atlas_ROIs.csv'), 
            skiprows=1, delimiter=',')
        self.uniqlabels = np.unique(rois_aparc[:,-1])
        cen = np.vstack([rois_aparc[:,:3],lh_coords,rh_coords]).mean(0)
        coords = rois_aparc[:,:3].copy()
        coords[:,:2] =- coords[:,:2]
        cens = np.asarray([coords[rois_aparc[:,-1]==l].mean(0) for l in self.uniqlabels])
        
        ### create subcortical surfaces expl
        lut_path = os.path.join(SRC_DIR, 'FreeSurferColorLUT.txt')
        lut_file = open(lut_path)
        self._lut = dict()
        for l in lut_file.readlines():
            if len(l)>4 and l[0]!='#':
                l = l.split()
                self._lut[int(l[0])] = (l[1],tuple(float(c)/255. for c in l[2:5]))
        lut_file.close()

        giis=[nibabel.load(glob.glob(os.path.join(SRC_DIR,'hcp_view_rois_surfs/%d.gii'%l))[0]) for l in self.uniqlabels]

        self._rois_surfaces = []
#        tr = np.asarray([[-1,0,0],[0,0,1],[0,-1,0]])
        for l,g,cc in zip(self.uniqlabels, giis,cens):
            roi_name = self._lut[l][0]
            
            surf_coords = g.darrays[0].data
            surf = mlab.triangular_mesh(surf_coords[:,0], surf_coords[:,1], surf_coords[:,2], g.darrays[1].data,
                                        opacity=.999)
                                        #opacity=SUBCTX_OPACITY)

            #surf.actor.property.color = self._lut[l][1]
            #surf.actor.mapper.scalar_visibility = False
            fakelut = np.asarray([self._lut[l][1]+(SUBCTX_OPACITY,)]*2)*255
            surf.module_manager.scalar_lut_manager.lut.table = fakelut
            surf.actor.actor.position = custom_shifts[roi_name]
            surf.name = roi_name
            self._rois_surfaces.append(surf)

            coords[rois_aparc[:,-1]==l] += custom_shifts[roi_name]

        self._pts = mlab.points3d(
            coords[:,0], coords[:,1], coords[:,2],
            mode='cube', scale_factor=1)

        self._scene.background = (0.0, 0.0, 0.0)
        self._pts.module_manager.scalar_lut_manager.lut.table = DEFAULT_LUT_TABLE * [1,1,1,.9]
#        self._pts.module_manager.scalar_lut_manager.lut_mode = lut_mode
#        self._pts.module_manager.scalar_lut_manager.reverse_lut = lut_reverse
        self._pts.module_manager.scalar_lut_manager.use_default_range = False
        self._pts.module_manager.scalar_lut_manager.lut.nan_color = [ 0.5,  0.5,  0.5,  0]
        self._pts.glyph.glyph.clamping = True
        self._pts.glyph.scale_mode = 'data_scaling_off'
        #self._pts.actor.property.opacity = 1 # make nan or zero invisible
        self._pts.actor.property.opacity = .999 # make nan or zero invisible

        self._scene.disable_render = False

        #self._scene.interactor.add_observer('KeyPressEvent', self.key_function)

    """
    def key_function(self, vtk_obj, event):
        self.vtk_obj=vtk_obj
        self.event=event
        self.keycode = vtk_obj.GetKeyCode()
    """
        
    def set_range(self, scalar_range):
        self._scalar_range = np.asarray(scalar_range)
        self._scene.disable_render = True
        self._pts.module_manager.scalar_lut_manager.data_range = self._scalar_range
        for surf in [self._lh_surf, self._rh_surf]:
            surf.module_manager.scalar_lut_manager.data_range = self._scalar_range
        self._pts.glyph.glyph.range = self._scalar_range
        self._pts.glyph.glyph.scale_factor = 1.0+1/float(self._scalar_range[1])
        #self._pts.glyph.glyph.clamping = True
        self._scene.disable_render = False

        
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

    def next(self):
        self.set_data_idx(self._data_idx+1)

    def prev(self):
        self.set_data_idx(self._data_idx-1)

    def _update(self, data):
        self._scene.disable_render = True
        lh_nverts = len(self._lh_surf.mlab_source.scalars)
        rh_nverts = len(self._rh_surf.mlab_source.scalars)
        self._lh_surf.mlab_source.scalars = data[:lh_nverts]
        self._rh_surf.mlab_source.scalars = data[lh_nverts:lh_nverts+rh_nverts]
        self._pts.mlab_source.scalars = data[lh_nverts+rh_nverts:]
        self.set_range(self._scalar_range)
        self._scene.disable_render = False
                
        
    def montage_screenshot(self,zoom=1.5,horiz=True):
        self._scene.disable_render = True
        # hide subcortical
        subcortical_objects = [self._pts]+self._rois_surfaces
        for el in subcortical_objects:
            el.visible = False
        self._scene.disable_render = False
        ## screenshot
        self._scene.parallel_projection = True
        self._scene.x_minus_view()
        self._scene.camera.view_up=[0,1,0]
        self._scene.camera.zoom(zoom*.80)
        self._scene.render()
        lh_lat = mlab.screenshot()

        #l,r=np.where(np.abs(np.diff(lh_lat.sum(-1).astype(np.int),1,0)).sum(0))[0][[0,-1]]+[-10,10]
        l,r=250,750
        lh_lat = lh_lat[:,l:r]
        self._scene.x_plus_view()
        self._scene.camera.view_up=[0,1,0]
        self._scene.camera.zoom(zoom*.80)
        self._scene.render()
        rh_lat = mlab.screenshot()
        #l,r=np.where(np.abs(np.diff(rh_lat.sum(-1).astype(np.int),1,0)).sum(0))[0][[0,-1]]+[-10,10]
        rh_lat = rh_lat[:,l:r]

        self._scene.disable_render = True
        for el in subcortical_objects:
            el.visible = True
        angle_medial_view = 30
        for surf in [self._lh_surf, self._lh_bg_surf]:
            surf.actor.actor.rotate_y(-angle_medial_view)
        for surf in [self._rh_surf, self._rh_bg_surf]:
            surf.actor.actor.rotate_y(angle_medial_view)
        
        self._scene.disable_render = False

        self._scene.z_plus_view()
        self._scene.camera.zoom(zoom*.95)
        self._scene.render()
        lrh_sup = mlab.screenshot()

        for surf in [self._lh_surf, self._lh_bg_surf]:
            surf.actor.actor.rotate_y(angle_medial_view)
        for surf in [self._rh_surf, self._rh_bg_surf]:
            surf.actor.actor.rotate_y(-angle_medial_view)

        self._scene.parallel_projection = False
        if horiz:
            montage = np.hstack([lh_lat,lrh_sup,rh_lat])
        else:
            montage = np.vstack([lrh_sup[125:800],np.hstack([lh_lat,rh_lat])[175:800]])
        return montage

        

def plot_montage(image, color_range, cmap=DEFAULT_LUT, boundaries=None, horiz=True):

    if horiz:
        figsize=(23,11)
    else:
        figsize=(10.6,14.375)
        
    fig, axes = pyplot.subplots(
        1,2,
        gridspec_kw=dict(width_ratios=[.95,.02], left=0, wspace=0),
        figsize=figsize,
    )
    axes[0].imshow(image)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    cbar = pyplot.matplotlib.colorbar.ColorbarBase(
        axes[1],
        cmap=cmap,
        norm=pyplot.matplotlib.colors.Normalize(vmin=0, vmax=cmap.N),
        boundaries=boundaries,
        orientation='vertical',
        ticks=np.linspace(0,cmap.N,color_range[1]-color_range[0]+1))
    axes[1].set_xticks([])
    axes[1].set_yticklabels(np.arange(color_range[0],color_range[1]+1),size=18)
    pyplot.subplots_adjust(left=.02, right=.95, top=.98, bottom=0.02) 
    return fig, axes, cbar

#cbar.set_ticks(np.linspace(0,1,5))
#cbar.set_ticklabels(np.arange(-20,21,5))
