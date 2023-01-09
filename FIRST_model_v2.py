'''
The FIRST model code.
Author: Shuaijun Liu
Date: 2022/07/23
Email: liushuaijun@mail.bnu.edu.cn
==============================================================================
Modified by Yuxuan YE
Update Feat:
    1. Remove the deprecated `scipy.misc.imresize()` function, using `PIL.Image.fromarray().resize()` instead.
    2. Add a `arr_resize()` function to resize an array.
    3. Export predicted F2 tiff file as `uint16` datatype (Not "uint19").
    4. Deleted packages that do not need to be imported.
    5. Correct the spelling of variables.
    6. Support parallel processing of multiple computing tasks.
Date: 2023/01/09
Email: yuxuanye145@gmail.com
==============================================================================
'''

import numpy as np
from osgeo import gdal
from sklearn.cross_decomposition import PLSRegression
from PIL import Image
from pathos.pools import ProcessPool
import os
import glob
# import matplotlib.pyplot as plt


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


def arr_resize(arr, w, h):
    # print(arr.min(), arr.max())
    # return np.array(Image.fromarray(arr.astype('uint8')).resize([w, h], 0)) / 255 * (arr.max() - arr.min()) + arr.min()
    return np.array(Image.fromarray(arr).resize([w, h], 0))


def resizeC2c(C, ratio):
	CH, W, H = C.shape
	Cc = np.zeros((CH, W//ratio, H//ratio))
	for i in range(CH):
		Cc[i] = arr_resize(C[i], W//ratio, H//ratio).astype(np.uint16)
	return Cc


# Pixel-based PLSR model
def PLS_Windows(C1_Csize, C2_Csize, F1, windows_size , n_components):
	pls2 = PLSRegression(copy=True, max_iter=800, n_components= n_components, scale=True, tol=1e-10)
	c, w, h = C1_Csize.shape
	C, W, H = F1.shape
    
	out_a = np.zeros((c, c, w, h))
	out_A = np.zeros((C, C, W, H))
	F2_Pre = np.zeros(F1.shape)
    
	out_C1_Mean = np.zeros((c, w, h))
	out_C1_Std = np.zeros((c, w, h))
	out_C2_Mean = np.zeros((c, w, h))
    
	out_F1_Mean = np.zeros((C, W, H))
	out_F1_Std = np.zeros((C, W, H))
	out_F2_Mean = np.zeros((C, W, H))
	
	for i in range(w):
		for j in range(h):
			left = max([0, i - windows_size])
			right = min([w - 1, i + windows_size])
			up = max([0, j - windows_size])
			down = min([h - 1, j + windows_size])
			getmodel_PLS = pls2.fit(C1_Csize[:, left:right, up:down].transpose((1, 2, 0)).reshape((-1, c)),
								  C2_Csize[:, left:right, up:down].transpose((1, 2, 0)).reshape((-1, c)))
			c1_mean, C1_std = C1_Csize[:, left:right, up:down].reshape((c, -1)).mean(axis=1), C1_Csize[:, left:right,up:down].reshape((c, -1)).std(axis=1)
			c2_mean = C2_Csize[:, left:right, up:down].reshape((c, -1)).mean(axis=1)
			# c2_std = C2_Csize[:, left:right, up:down].reshape((c, -1)).std(axis=1)
			coef_ = getmodel_PLS.coef_
			out_a[:, :, i, j] = coef_
			# print('a:', out_a[:, :, i, j])
			out_C1_Mean[:, i, j] = c1_mean
			out_C1_Std[:, i, j] = C1_std
			out_C2_Mean[:, i, j] = c2_mean
    
	for i in range(c):
		out_F1_Mean[i] = arr_resize(out_C1_Mean[i], W, H)
		out_F1_Std[i] = arr_resize(out_C1_Std[i], W, H)
		out_F2_Mean[i] = arr_resize(out_C2_Mean[i], W, H)
		for j in range(c):
			out_A[i, j, :, :] = arr_resize(out_a[i, j, :, :], W, H)
    
	for i in range(W):
		for j in range(H):
			XX = (F1[:, i, j] - out_F1_Mean[:, i, j]) / out_F1_Std[:, i, j]
			YY = (np.dot(XX, out_A[:, :, i, j])) + out_F2_Mean[:, i, j]
			F2_Pre[:, i, j] = YY
	return F2_Pre


# Image-based PLSR model
def PLS_WHOLE(C1, C2, F1, components):
	c, w, h = C1.shape
	pls2 = PLSRegression(copy=True, max_iter=800, n_components=components, scale=True,
						 tol=1e-10)
	getmodel_PLS = pls2.fit(C1.transpose((1, 2, 0)).reshape((-1, c)),
			 C2.transpose((1, 2, 0)).reshape((-1, c)))
	coef_ = getmodel_PLS.coef_
	l = getmodel_PLS.x_loadings_
	# u = getmodel_PLS.x_scores_
	n = getmodel_PLS.y_loadings_
	coef_u = l*n.transpose()
	dif = coef_u - coef_
	print(dif)
# 	color_list = ["r", "y", "b", "g", "b*-", "g*-"]
# 	for i in range(c):
# 		plt.plot(range(c), coef_[:, i], color_list[i], label='Plot'+ str(i))
# 	plt.legend()
# 	plt.show()
	a_test = pls2.predict(F1.transpose((1, 2, 0)).reshape((-1, c))).transpose((1, 0)).reshape(F1.shape)
	return a_test


def FIRST(C1, C2, F1, outputf_name, windows_size, n_components, ratio, whole = True):
    try:
        if whole == True:
            result = PLS_WHOLE(C1, C2, F1, n_components)
        else:
            result = PLS_Windows(resizeC2c(C1, ratio), resizeC2c(C2, ratio), F1, windows_size = windows_size, n_components = n_components)
        imsave(result, outputf_name, 'uint16')
        return 1
    except:
        return 2


def main_FIRST_original():
    # Adjustment of window size based on experimental data，we recommend a half window size of 4 to 6 for coarse's size image
    windows_size = 5
    # Adjustment of the upper limit of the components for PLSR. It should be less than or equal to the number of bands.
    band_components = 6
    # Ratio between coarse images and fine images
    ratio = 20
    # Whether to use the image-based PLSR model
    is_whole = False
    
    # The path and file name of the input image, and the path of the output image
    C1 = imread('/Users/yonniye/Desktop/fusion/data/group_1/C1_1')
    C2 = imread('/Users/yonniye/Desktop/fusion/data/group_1/C2_1')
    F1 = imread('/Users/yonniye/Desktop/fusion/data/group_1/F1_1')
    outputf_name = '/Users/yonniye/Desktop/fusion/data/group_1/F2_1_FIRST.tif'
    FIRST(C1, C2, F1, outputf_name, windows_size, band_components, ratio, is_whole)


def main_FIRST_from_path(dpath):
    # Parameters
    windows_size = 5
    band_components = 6
    ratio = 20
    is_whole = False
    
    group_num = dpath.split("/")[-1].split("_")[1]
    f1_data = imread(os.path.join(dpath, f"F1_{group_num}"))
    c1_data = imread(os.path.join(dpath, f"C1_{group_num}"))
    c2_data = imread(os.path.join(dpath, f"C2_{group_num}"))
    out_path = os.path.join(dpath, f"F2_{group_num}_FIRST.tif")
    if not os.path.exists(out_path):
        return FIRST(c1_data, c2_data, f1_data, out_path, windows_size, band_components, ratio, is_whole)
    else:
        return 3
    

if __name__ == "__main__":
    # main_FIRST_original()
    root_path = sorted(glob.glob("/Users/yonniye/Desktop/fusion/data/group_*/*"))
    with ProcessPool(5) as pool:
        for r in pool.imap(main_FIRST_from_path, root_path):
            print(r)