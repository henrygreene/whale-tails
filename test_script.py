import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import time
import cv2
import sklearn
import torch
from whale_image_proc import *  # Custom made image processing functions


# data_dir = "D:/Users/Craig/Documents/Large Data Files/Kaggle Whale Data/"
data_dir = "C:/Users/Craig/Large Data Sets/"
train_img_dir = data_dir + "train/train/"

train_df = pd.read_csv(data_dir + "train.csv")

t = time.time()
ind = 0
min_shape = [np.inf, np.inf]
imgsarr = np.empty([0, 0, 0])
npix = 256
imgs_train = []
for i in train_df.index:
    img = Image.open(train_img_dir + train_df.Image[i])
    imgsqr = img.resize((npix,npix),Image.ANTIALIAS)
    if imgsqr.mode != 'L':
        imgsqr = imgsqr.convert(mode='L')
    imgs_train.append(imgsqr)
    img.close()
    if i > 20:
        break
elapse = time.time()-t
print(elapse)

# t = time.time()
# U, s, V = np.linalg.svd(imgsarr, full_matrices=False)
# elapse = time.time()-t
# print(elapse)
# test_proj = np.array(np.mat(V[0:250, :]).T*(np.mat(V[0:250, :])*np.mat(imgsarr[0, :]).T))
# img = Image.open(train_img_dir + train_df.Image[11])
# imgvec = np.array(img)

# whale_id = train_df.Id[6]
# img1 = np.asarray(imgs_train[6])
# img1_edges = np.copy(img1)
# cv2.Canny(img1, 500, 1500, img1_edges, 5, True)
# img2 = np.asarray(imgs_train[701])
# img2_edges = np.copy(img2)
# cv2.Canny(img2, 500, 1500, img2_edges, 5, True)
# img3 = pixel_cluster(img2, 256, 5)
# img3_edges = np.copy(img3)
# cv2.Canny(img3, 500, 1500, img3_edges, 5, True)
#
#
# fig, axes = plt.subplots(3, 2, num=1, clear=True)
# axes[0, 0].imshow(img1)
# axes[0, 1].imshow(img1_edges)
# axes[1, 0].imshow(img2)
# axes[1, 1].imshow(img2_edges)
# axes[2, 0].imshow(img3)
# axes[2, 1].imshow(img3_edges)
# fig.suptitle("Whale Id: {}".format(whale_id))
#
start = time.time()
for k in range(10):
    pixel_cluster_gauss(np.asarray(imgs_train[k]), 10, iters=100, prec_thresh=1e-6, iters_inner=25, smooth=3)
    plt.savefig("../six_gauss_fit_img{}".format(k))
print("Total Time: {}".format(time.time()-start))


# TODO Start machine learning framework
