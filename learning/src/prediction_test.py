#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import cv2
import math

file = "/home/tony/Downloads/images/frame_58.jpg"

img = cv2.imread(file)

img_final_height = 100
img_final_width = 200

new_img = img[img.shape[0]/3:img.shape[0]*2/3, :, :]
# = cv2.cvtColor(new_img[:, :, :], cv2.COLOR_RGB2GRAY)
new_img = cv2.resize( new_img, (img_final_width, img_final_height) ) 
new_img = new_img.astype(float)
new_img = new_img / 255

new_img = np.reshape(new_img, (1, -1))
print("Shape", new_img.shape)
print("=====================================")

tf.device('/device:GPU:0')

# Import the TF graph
filename = "../learned_models/tensorflow_logs/graph/graph.pb"

#for i in tf.get_default_graph().get_operations():
#    print i.name

sess = tf.Session()
filename = "../learned_models/tensorflow_logs/train-1400.meta"
saver = tf.train.import_meta_graph(filename)
saver.restore(sess, '../learned_models/tensorflow_logs/train-1400')

#print(sess.run('ConvNet/fc_layer_2/bias:0'))


img_tensor = sess.graph.get_tensor_by_name('x:0')
output_tensor = sess.graph.get_tensor_by_name('ConvNet/fc_layer_2/BiasAdd:0')

pred = ret = sess.run(output_tensor, {img_tensor: new_img})
left = pred[0][0]
right = pred[0][1]
print(str(left)+"_" + str(right))

v = 3
omega = (left-right)*2.8

rad = omega*math.pi/2. # [-1.57~1.57]
rad = rad - math.pi/2. # rotae for correct direction
radius = 60
alpha = 0.3

v_length = 150

x = v_length*math.cos(rad)
y = v_length*math.sin(rad)
center = (int(img.shape[1]/2), int(img.shape[0]))


cv2.circle(img, (center[0], center[1]), v_length+30, (0, 0 ,255), 8)
cv2.arrowedLine(img, center, (int(center[0]+x), int(center[1]+y)), (0, 0, 255), 8)

cv2.imwrite(str(left)+"_" + str(right) + ".jpg", img)
