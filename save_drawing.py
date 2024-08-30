import pygame
import numpy as np

def draw(window_size,screen,matrix,draw_size,mouse_loc,radius):
    x,y=int(draw_size[0]*mouse_loc[0]/window_size[0]),int(draw_size[1]*mouse_loc[1]/window_size[1]) #coordinate of the mouse in the matrix
    add_circle(x,y,radius,matrix,draw_size)
    #matrix[x,y]=[255,255,255]
    
    
pattern=[0,[[1]],[[0.1,0.4,0.1],[0.4,1,0.4],[0.1,0.4,0.1]],[[0.1,0.2,0.5,0.2,0.1],[0.2,0.5,0.7,0.5,0.2],[0.5,0.7,1,0.7,0.5],[0.2,0.5,0.7,0.5,0.2],[0.1,0.2,0.5,0.2,0.1]]]

def add_circle(x,y,radius,matrix,draw_size):
    radius=max(1, min(3, radius))
    rangee=2*radius-1
    patt=pattern[radius]
    for i in range(rangee):
        for j in range(rangee):
            if x-radius+1+i>=0 and y-radius+1+j>=0 and x-radius+1+i<draw_size[0] and y-radius+1+j<draw_size[1]:
                matrix[x-radius+1+i,y-radius+1+j]=bound01(matrix[x-radius+1+i,y-radius+1+j]+patt[i][j])
                #matrix[int(x-radius+i),int(y-radius+j)]=add_vect(matrix[int(x-radius+i),int(y-radius+j)],patt[i][j])
                

def bound01(e):
    return max(0, min(1, e))
                
    