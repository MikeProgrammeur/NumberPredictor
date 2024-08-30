import pygame
import numpy as np

def aff(canva_size,screen,matrix,draw_size):
    rect_width=canva_size[0]/draw_size[0]
    rect_height=canva_size[1]/draw_size[1]
    x,y=0,0
    for i in range(draw_size[0]):
        y=0
        for j in range(draw_size[1]):
            #print(x,y)
            pygame.draw.rect(screen,matrix[i,j]*np.array([255,255,255]),(x,y,rect_width,rect_height))
            y=int(y+rect_height)
        x=int(x+rect_width)
            
        