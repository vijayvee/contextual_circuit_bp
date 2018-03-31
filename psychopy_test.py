#!/usr/bin/python
from psychopy import visual, event, monitors
import numpy as np
from tqdm import tqdm
"""Code to create the snakes dataset for association fields"""
pointsInLine=[]

def create_window(window_size, monitor):
    #create a window
    mywin = visual.Window([window_size[0], window_size[1]] ,color=(-1,-1,-1), monitor=monitor, units="deg")
    return mywin

def draw_line(win, start, end, lineWidth=3.):
    #function to draw a line
    ln = visual.Line(win=win,start=start,end=end,lineWidth=lineWidth,lineColor=(1,-1,-1))
    ln.draw()

def draw_grating_stim(win, pos, size,ori=0):
    stim = visual.GratingStim(win=win, pos=pos, size=(0.3,1.0), tex='sin', mask='cross', units='deg', sf=1.5, ori=ori)
    return stim

def draw_circle(win,radius,lineWidth=3.,edges=32):
    crcl = visual.Circle(win=win,radius=radius,fillColor=(-1,-1,-1),lineWidth=lineWidth,edges=edges,lineColor=(-1,-1,-1))
    crcl.draw()
    return crcl

def shear(points,shearX):
    new_points = np.zeros(points.shape)
    shear_matrix = np.array([[1,0],[shearX,1]])
    new_points = points.dot(shear_matrix)
    return new_points

def draw_lines_row(win, circle, positions, lineWidth=0.13):
    pos_x, pos_y = positions.shape
    for i in tqdm(range(pos_x),desc='Rows'):
        for j in tqdm(range(pos_y),desc='Cols'):
            pos = positions[i,j]
            if circle.contains(pos):
        	    if pos[0]==pos[i] and np.abs(pos[0])<2:
                        ln = visual.Line(win, pos=pos, size=lineWidth, ori=0,lineColor=(1,0,0))
        	    else:
                        ln = visual.Line(win, pos=pos, size=lineWidth, ori=np.random.randint(0,180))
                    ln.draw()

def draw_fixation(win, x, y):
    fixation = visual.GratingStim(win=win, size=0.04, pos=[x,y], sf=0, color=(1,-1,-1))
    fixation.draw()

def main():
    mon = 'testMonitor'
#    mon.setDistance(35)
    win = create_window([256,256],monitor=mon)
    print "Created window"
    circle = draw_circle(win=win,radius=3)
    print "Central circle drawn"
    pos = []
    for i in np.arange(-4,4,0.5):
        for j in np.arange(-4,4,0.25):
            pos.append([j,i])
    import ipdb; ipdb.set_trace()
    positions = np.array(pos)
    draw_lines_row(win, circle, positions)
    draw_fixation(win, 0, 0)
    win.update()
    event.waitKeys()
    win.getMovieFrame()
    win.saveMovieFrames("sample.png")
    win.close()

if __name__=="__main__":
    main()
