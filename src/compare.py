import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL
import numpy as np
import scipy.io as sio

def compare_matrix(encoder_output, class_id):
  gd_img=sio.loadmat('../data/Samson/GroundTruth/end3.mat')
  gd_grid=np.reshape(gd_img['A'][class_id], newshape=(95, 95))

  print("Accuracy: ", 100*(1.0-((encoder_output - gd_grid)**2).mean(axis=None)))