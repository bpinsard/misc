import cv2
import numpy as np
import scipy.fftpack


def scramble(f):
    cap=cv2.VideoCapture(f)
    
    perm = None
    noise = None

    fourcc = cv2.cv.CV_FOURCC(*'PIM1')
    out_fname = '.'.join(f.split('.')[:-1])+'_scrambled.mpg'
    #print out_fname
    out = None
    
    i=0
    ret, frame = cap.read()

    perm = np.random.permutation(np.arange(frame[:,:,0].size))
    noise = np.random.vonmises(mu=0,kappa=.05,size=frame.shape[:2])
    out = cv2.VideoWriter(out_fname, fourcc, cap.get(cv2.cv.CV_CAP_PROP_FPS), frame.shape[:2][::-1])

    while(ret):
        print i
        i += 1

#        if i>10:
#            break
            
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float)/255.
#        return hsv
        ft = [scipy.fftpack.fft2(hsv[...,c]) for c in range(3)]
#        scph = [np.angle(ft[c]).ravel()[perm].reshape(ft[c].shape) for c in range(3)]
#        scph = [np.angle(ft[c])+np.angle(ft[c]).ravel()[perm].reshape(ft[c].shape) for c in range(3)]
#        scph = [np.angle(ft[c]) for c in range(3)]
        scph = [np.mod(np.angle(ft[c])+noise+3*np.pi,2*np.pi)-np.pi for c in range(3)]
        ftsc = [np.abs(ft[c])*np.exp(scph[c]*1j) for c in range(3)]
        ift = [scipy.fftpack.ifft2(ftsc[c]) for c in range(3)]
#        print [(np.abs(c[...,np.newaxis]).min(),np.abs(c[...,np.newaxis]).max()) for c in ift]
        oframe = (np.dstack([np.abs(c[...,np.newaxis]) for c in ift]))
        maxs = np.apply_over_axes(np.max,oframe,[0,1])
#        print maxs.ravel()
        oframe[...,maxs.ravel()>=1] /= maxs[maxs>=1]
#        print oframe.max()
        bgr = cv2.cvtColor((oframe*255).astype(np.uint8), cv2.COLOR_LAB2BGR)
#        print frame.shape,frame.dtype
#        print bgr.shape, bgr.dtype
        out.write(bgr)
#        out.write(frame)
        del hsv, ft, ftsc, ift, oframe, bgr
        ret, frame = cap.read()

    cap.release()
    del out
    cv2.destroyAllWindows()
    
    
