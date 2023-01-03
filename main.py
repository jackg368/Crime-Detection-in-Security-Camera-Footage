import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

from model import define_compile_model
from utils import multiclass_roc_auc_score
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
  # Set params
  SEED = 12
  IMG_HEIGHT = 64
  IMG_WIDTH = 64
  BATCH_SIZE = 64
  EPOCHS = 1
  LR =  0.00003
  NUM_CLASSES = 14
  CLASS_LABELS = ['Abuse','Arrest','Arson','Assault','Burglary','Explosion','Fighting',"Normal",'RoadAccidents','Robbery','Shooting','Shoplifting','Stealing','Vandalism']

  # Load data
  train_dir = "../input/ucf-crime-dataset/Train"
  test_dir = "../input/ucf-crime-dataset/Test"
  preprocess_fun = tf.keras.applications.densenet.preprocess_input

  train_datagen = ImageDataGenerator(horizontal_flip=True,
                                    width_shift_range=0.1,
                                    height_shift_range=0.05,
                                    rescale = 1./255,
                                    preprocessing_function=preprocess_fun
                                    )
  test_datagen = ImageDataGenerator(rescale = 1./255,
                                    preprocessing_function=preprocess_fun
                                  )
  start = time.time()
  train_generator = train_datagen.flow_from_directory(directory = train_dir,
                                                      target_size = (IMG_HEIGHT ,IMG_WIDTH),
                                                      batch_size = BATCH_SIZE,
                                                      shuffle  = True , 
                                                      color_mode = "rgb",
                                                      class_mode = "categorical",
                                                      seed = SEED
                                                    )
  test_generator = test_datagen.flow_from_directory(directory = test_dir,
                                                    target_size = (IMG_HEIGHT ,IMG_WIDTH),
                                                      batch_size = BATCH_SIZE,
                                                      shuffle  = False , 
                                                      color_mode = "rgb",
                                                      class_mode = "categorical",
                                                      seed = SEED
                                                    )
  print(f'Loading finished in {time.time()-start:.2f}s')

  # Build model
  model = define_compile_model(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES, LR)
  model.summary()

  # Train model
  history = model.fit(x = train_generator,validation_data=test_generator,epochs = EPOCHS)

  # Evaluate model
  preds = model.predict(test_generator)
  y_test = test_generator.classes
  fig, c_ax = plt.subplots(1,1, figsize = (15,8))
  print('ROC AUC score:', multiclass_roc_auc_score(c_ax, y_test , preds, CLASS_LABELS, average = "micro"))
  plt.xlabel('FALSE POSITIVE RATE', fontsize=18)
  plt.ylabel('TRUE POSITIVE RATE', fontsize=16)
  plt.legend(fontsize = 11.5)
  plt.show()
