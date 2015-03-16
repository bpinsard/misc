from mayavi.scripts import mayavi2
from mayavi.sources.array_source import ArraySource
import traits.api as traits

from traitsui.api import View, Item, HGroup, Group

import matplotlib.pyplot as plt

from mayavi import mlab
from pyface.api import GUI
from traits.api import HasTraits, Button, Instance
from nibabel.affines import apply_affine
from threading import Thread
import numpy as np
from time import sleep

class MocoJob(Thread):
    def __init__(self, algo, stack, **kwargs):
        Thread.__init__(self, **kwargs)
        self.controller = None
        self.algo = algo
        self.stack = stack
        self.registered_slabs = []
        self.pause = False
        
    def run(self):
        print 'run'
        old_fr=-1
        it = self.algo.process(self.stack,yield_raw=True)
        stack_has_data = True
        fr,slab,reg,data = it.next()
        while True:
            while not self.pause and stack_has_data:
                print ('volume%d slab:['+('%d,'*len(slab))+']')%((fr,)+tuple(slab))
                self.registered_slabs.append((fr,slab,reg,data))
                self.controller.notify_changes()
                try:
                    fr,slab,reg,data = it.next()
                except StopIteration:
                    stack_has_data = False
            if not stack_has_data:
                break
            sleep(2)
            

class MotionView(object):

#    data_src = Instance(ArraySource)

    def __init__(self,stack,init_reg,*args,**kwargs):
        self.stack_idx = dict([[s[1][0],i] for i,s in enumerate(stack._slabs)])
        bounds = [[0,stack._shape[0]],[0,stack._shape[1]]]
        points = np.asarray([(x,y,0) for x in bounds[0] for y in bounds[1]])
        zz = np.asarray([0,0,1])
        tris_plane = np.asarray([[0,1,2],[1,2,3]])
        self.points = np.array([[points+zz*s  for s in slab[1]] for slab in stack._slabs])
        self.coords = apply_affine(init_reg,self.points)
        tris = np.vstack([tris_plane+4*(i+si*len(slab[1])) for si,slab in enumerate(stack._slabs) for i,s in enumerate(slab[1])])
        colors = np.hstack([np.zeros(len(slab[1])*4)+i for i,slab in enumerate(stack._slabs)])
#        self.slices = mlab.triangular_mesh(
#            self.coords[...,0].ravel(),self.coords[...,1].ravel(),self.coords[...,2].ravel(),tris,
#            scalars = colors, scale_mode='none')
        
#        self.ipw = mlab.pipeline.image_plane_widget(
#            self.data_src,
#            )
        plt.ion()
        self.ntimes = 0
        self.slab_nslices = len(stack._slabs[0][1])
        self.nslabs = len(stack._slabs)
        fig, ax = plt.subplots(1,self.slab_nslices,squeeze=False)
        self.slab_ims = []
        for si in range(self.slab_nslices):
            self.slab_ims.append(ax[0,si].matshow(np.zeros(stack._shape[:2]),cmap=plt.get_cmap('gray'),vmin=0,vmax=2000))
        self.motion_fig,self.motion_plot = plt.subplots()
        huge_num = 1000
        self.motion_plot.bar(
            left = np.arange(0,huge_num*self.nslabs,self.nslabs)-.5,
            height=200*np.ones(np.ceil(huge_num)),
            bottom=-100, width=self.nslabs,color=['b','g'], 
            linewidth=0,alpha=.2)
#        self.motion_plot_bg = fig.canvas.copy_from_bbox(self.motion_plot.bbox)
        self.motion_plot_lines = self.motion_plot.plot([0],[[0]*6],'+-')
        self.motion_plot.set_ylim(-.1,.1)
        self.motion_plot.set_xlim(-.5,100)

    def update_slab(self, frame, slab, reg, data):
        print 'update coords'
        """
        stidx = self.stack_idx[slab[0]]
        self.coords[stidx] = apply_affine(reg, self.points[stidx])
        print self.slices.mlab_source.x.max()
        self.slices.mlab_source.x = self.coords[...,0].ravel()
        self.slices.mlab_source.y = self.coords[...,1].ravel()
        self.slices.mlab_source.z = self.coords[...,2].ravel()
        GUI.invoke_later(self.slices.mlab_source.update)
        """
        for si,sl in enumerate(slab):
            print 'update image'
            self.slab_ims[si].set_array(data[...,si])
        plt.draw()


    def update_motion(self, motion):
        if motion.shape[0] == self.ntimes:
            return
        self.ntimes = motion.shape[0]
        nmotion=motion-motion[0]
#        self.motion_fig.canvas.restore_region(self.motion_plot_bg)
        for i,l in enumerate(self.motion_plot_lines):
            l.set_data(np.arange(self.ntimes),nmotion[:,i])
        plt.draw()
#            self.motion_plot.draw_artist(l)
#        self.motion_plot.set_ylim(np.round(nmotion.min()-1),np.round(nmotion.max()+1))
#        self.motion_fig.canvas.blit(self.motion_plot.bbox)
#        self.motion_plot.set_xlim(-.5,motion.shape[0]+10)

class Controller(traits.HasTraits):
    
    min_slab = traits.Int(0)
    max_slab = traits.Int(1E9)
    slab = traits.Range(low='min_slab',high='max_slab',value=0,mode='spinner',exclude_high=True)

    apply_registration = traits.Bool()

    pause = traits.Bool()

    view = View(
        Group(Item(name='slab'),Item(name='apply_registration'),Item(name='pause'))
        )    

    def __init__(self,view,job,*args,**kwargs):
        HasTraits.__init__(self,*args,**kwargs)
        self.view = view
        self.job = job
        self.max_slab=0

    @traits.on_trait_change("pause")
    def job_pause(self,value):
        self.job.pause = value

    @traits.on_trait_change("slab,apply_registration")
    def update_slab(self):
        ## switch between the register of the previous and the current slab
        if self.apply_registration or self.slab>0 and len(self.job.registered_slabs)>0:
            sl = self.job.registered_slabs[self.slab]
            sl_reg = self.job.registered_slabs[self.slab+int(self.apply_registration)-1]
            self.view.update_slab(sl[0],sl[1],sl_reg[2],sl[3])
    def notify_changes(self):
        self.max_slab = len(self.job.registered_slabs)
        self.view.update_motion(np.c_[self.job.algo.filtered_state_means])
        
def init(algo,stack):
    stack._init_dataset()
    job = MocoJob(algo, stack)
    view = MotionView(stack, algo.init_reg.dot(stack._affine))
    controller = Controller(view,job)
    job.controller = controller
    controller.edit_traits()
    controller.max_slab = 0
    job.start()

