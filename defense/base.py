import os,sys
import numpy as np
import torch


class defense(object):


    def __init__(self,):
        print(1)

    def add_arguments(parser):
        print('You need to rewrite this method for passing parameters')
    
    def set_result(self):
        print('You need to rewrite this method to load the attack result')
        
    def set_trainer(self):
        print('If you want to use standard trainer module, please rewrite this method')
    
    def set_logger(self):
        print('If you want to use standard logger, please rewrite this method')

    def denoising(self):
        print('this method does not have this function')

    def mitigation(self):
        print('this method does not have this function')

    def inhibition(self):
        print('this method does not have this function')
    
    def defense(self):
        print('this method does not have this function')
    
    def detect(self):
        print('this method does not have this function')

