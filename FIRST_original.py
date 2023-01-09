'''
The FIRST model code.
Author: Shuaijun Liu
Date: 2022/07/23
Email: liushuaijun@mail.bnu.edu.cn
'''
import numpy as np
import matplotlib.pyplot as plt
import gdal
import scipy.misc as m
from sklearn.cross_decomposition import PLSRegression
from sklearn.cross_decomposition import CCA
from tqdm import trange
import os
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_similarity

from sewar.full_ref import rmse
from sewar.full_ref import sam
from sewar.full_ref import ssim
from sewar import vifp      #视觉信息保真度
from sewar import d_lambda

from collections import Counter

# Multi-band image reading and writing functions
def imread(path, startX=0, startY=0, X=0, Y=0):
	ds = gdal.Open(path)
	im_width = ds.RasterYSize
	im_height = ds.RasterXSize

	if X != 0:
		im_height = X
	if Y != 0:
		im_width = Y

	img = np.array(ds.ReadAsArray(startX, startY, im_height, im_width))
	img = img.astype(np.int)
	return img
def imsave(img, path, Dtype):
	if len(img.shape) == 3:
		(n, h, w) = img.shape
	else:
		(h, w) = img.shape
		n = 1
	driver = gdal.GetDriverByName("GTiff")

	if Dtype == 'uint8':
		datatype = gdal.GDT_Byte
	elif Dtype == 'uint16':
		datatype = gdal.GDT_UInt16
	else:
		datatype = gdal.GDT_Float32
	dataset = driver.Create(path, w, h, n, datatype)
	if len(img.shape) == 3:
		for t in range(n):
			dataset.GetRasterBand(t + 1).WriteArray(img[t])
	else:
		dataset.GetRasterBand(1).WriteArray(img)

	del dataset

def resizeC2c(C, ratio):
	CH, W, H = C.shape
	Cc = np.zeros((CH, W//ratio, H//ratio))
	for i in range(CH):
		Cc[i, :, :] = m.imresize(C[i, :, :], (W//ratio, H//ratio), 'nearest') / 255 * (C[i, :, :].max() - C[i, :, :].min()) + C[i, :,:].min()
	return Cc

# Pixel-based PLSR model
def PLS_Windows(C1_Csize, C2_Csize, F1, winodws_size , n_components):
	# 最终版偏最小二乘的实现
	pls2 = PLSRegression(copy=True, max_iter=500, n_components= n_components, scale=True,
						 tol=1e-10)
	c, w, h = C1_Csize.shape
	C, W, H = F1.shape
	Nihe_Image_a = np.zeros((c, c, w, h))
	Nihe_Image_A = np.zeros((C, C, W, H))

	Nihe_Image_C1_Mean = np.zeros((c, w, h))
	Nihe_Image_C1_Std = np.zeros((c, w, h))
	Nihe_Image_C2_Mean = np.zeros((c, w, h))
	Nihe_Image_F1_Mean = np.zeros((C, W, H))
	Nihe_Image_F1_Std = np.zeros((C, W, H))
	Nihe_Image_F2_Mean = np.zeros((C, W, H))

	F2_Pre = np.zeros(F1.shape)
	for i in trange(w):
		for j in range(h):
			left = max([0, i - winodws_size])
			right = min([w - 1, i + winodws_size])
			up = max([0, j - winodws_size])
			down = min([h - 1, j + winodws_size])  # 搜索最相近曲线的窗口：20-20
			getmodel_PLS = pls2.fit(C1_Csize[:, left:right, up:down].transpose((1, 2, 0)).reshape((-1, c)),
								  C2_Csize[:, left:right, up:down].transpose((1, 2, 0)).reshape((-1, c)))
			c1_mean, C1_std = C1_Csize[:, left:right, up:down].reshape((c, -1)).mean(axis=1), C1_Csize[:, left:right,
																							  up:down].reshape(
				(c, -1)).std(axis=1)
			c2_mean = C2_Csize[:, left:right, up:down].reshape((c, -1)).mean(axis=1)
			c2_std = C2_Csize[:, left:right, up:down].reshape((c, -1)).std(axis=1)
			coef_ = getmodel_PLS.coef_
			Nihe_Image_a[:, :, i, j] = coef_
			#print('a:', Nihe_Image_a[:, :, i, j])
			Nihe_Image_C1_Mean[:, i, j] = c1_mean
			Nihe_Image_C1_Std[:, i, j] = C1_std
			Nihe_Image_C2_Mean[:, i, j] = c2_mean

	for i in range(c):
		Nihe_Image_F1_Mean[i, :, :] = m.imresize(Nihe_Image_C1_Mean[i, :, :], (W, H), 'nearest') / 255 * (
					Nihe_Image_C1_Mean[i, :, :].max() - Nihe_Image_C1_Mean[i, :, :].min()) + Nihe_Image_C1_Mean[i, :,
																							 :].min()
		Nihe_Image_F1_Std[i, :, :] = m.imresize(Nihe_Image_C1_Std[i, :, :], (W, H), 'nearest') / 255 * (
					Nihe_Image_C1_Std[i, :, :].max() - Nihe_Image_C1_Std[i, :, :].min()) + Nihe_Image_C1_Std[i, :,
																						   :].min()
		Nihe_Image_F2_Mean[i, :, :] = m.imresize(Nihe_Image_C2_Mean[i, :, :], (W, H), 'nearest') / 255 * (
					Nihe_Image_C2_Mean[i, :, :].max() - Nihe_Image_C2_Mean[i, :, :].min()) + Nihe_Image_C2_Mean[i, :,
																							 :].min()

		for j in range(c):
			Nihe_Image_A[i, j, :, :] = m.imresize(Nihe_Image_a[i, j, :, :], (W, H), 'nearest') / 255 * (
						Nihe_Image_a[i, j, :, :].max() - Nihe_Image_a[i, j, :, :].min()) + Nihe_Image_a[i, j, :,
																						   :].min()

	for i in trange(W):
		for j in range(H):
			XX = (F1[:, i, j] - Nihe_Image_F1_Mean[:, i, j]) / Nihe_Image_F1_Std[:, i, j]
			YY = (np.dot(XX, Nihe_Image_A[:, :, i, j])) + Nihe_Image_F2_Mean[:, i, j]
			F2_Pre[:, i, j] = YY
			# if np.sum(C1[:, i, j].reshape(1,-1) - C2[:, i, j].reshape(1,-1)) == 0:
			# 	F2_Pre[:, i, j] = F1[:, i, j]

	return  F2_Pre

# Image-based PLSR model
def PLS_WHOLE(C1, C2, F1, components):
	c, w, h = C1.shape
	pls2 = PLSRegression(copy=True, max_iter=500, n_components=components, scale=True,
						 tol=1e-10)
	getmodel_PLS = pls2.fit(C1.transpose((1, 2, 0)).reshape((-1, c)),
			 C2.transpose((1, 2, 0)).reshape((-1, c)))
	coef_ = getmodel_PLS.coef_
	l = getmodel_PLS.x_loadings_
	u = getmodel_PLS.x_scores_
	n = getmodel_PLS.y_loadings_
	coef_u = l*n.transpose()
	dif = coef_u - coef_
	print(dif)

	# color_list = ["r", "y", "b", "g", "b*-", "g*-"]
	# for i in range(c):
	# 	plt.plot(range(c), coef_[:, i], color_list[i], label='Plot'+ str(i))
	# plt.legend()
	# plt.show()
	a_test = pls2.predict(F1.transpose((1, 2, 0)).reshape((-1, c))).transpose((1, 0)).reshape(F1.shape)
	return  a_test

def FIRST(C1, C2, F1, outputf_name, winodws_size, n_components, ratio, whole = True):
    if whole == True:
        result = PLS_WHOLE(C1, C2, F1, components)
    else:
        result = PLS_Windows(resizeC2c(C1, ratio), resizeC2c(C2, ratio), F1, winodws_size = winodws_size, n_components = n_components)

    imsave(result, outputf_name, 'uint19')

def main_FIRST():

    # Adjustment of window size based on experimental data，we recommend a half window size of 4 to 6 for coarse's size image
    winodws_size = 50
    # Adjustment of the upper limit of the components for PLSR. It should be less than or equal to the number of bands.
    band_components = 3
    # Ratio between coarse images and fine images
    ratio = 30
    # Whether to use the image-based PLSR model
    whole = True
    # The path and file name of the input image, and the path of the output image
    C1 = imread(r'D:\DataRS\Evaluate_Fusion\True_Image2\M__2001280.tif')
    C2 = imread(r'D:\DataRS\Evaluate_Fusion\True_Image2\M__2001289.tif')
    F1 = imread(r'D:\DataRS\Evaluate_Fusion\True_Image2\L7_2001280.tif')
    outputf_name = r'D:\DataRS\Evaluate_Fusion\True_Image2\result.tif'
    FIRST(C1, C2, F1, outputf_name, winodws_size, components, ratio, whole)


if __name__ == "__main__":
    main_FIRST()