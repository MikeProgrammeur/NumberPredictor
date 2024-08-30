import numpy as np
import pygame

class Button:
    def __init__(self,color,text,location,dimension):
        self.__color=color
        self.__text=text
        self.__location=location
        self.__dimension=dimension
    
    def pressed(self,mouseloc,mousepressed):
        result = mousepressed and self.__location[0] < mouseloc[0] and self.__location[1] < mouseloc[1]
        result &= self.__location[0]+self.__dimension[0] > mouseloc[0]
        result &= self.__location[1]+self.__dimension[1] > mouseloc[1]
        return result
    
    def display(self,screen):
        pygame.draw.rect(screen,self.__color,(*self.__location,*self.__dimension))