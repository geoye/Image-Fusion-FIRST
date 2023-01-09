# Image-Fusion-FIRST
Python implementation and modification of the spatiotemporal Fusion Incorrporting Spectral autocorrelaTion (FIRST) model

The original code is downloaded from: http://www.chen-lab.club/?p=19855

Update Feats:
1. Remove the deprecated `scipy.misc.imresize()` function, using `PIL.Image.fromarray().resize()` instead.
2. Add a `arr_resize()` function to resize an array.
3. Export predicted F2 tiff file as `uint16` datatype (Not "uint19").
4. Deleted packages that do not need to be imported.
5. Correct the spelling of variables.
6. Support parallel processing of multiple computing tasks.

Note: If the `ConvergenceWarning` appears, please increase the size of the `max_iter` paramter of `PLSRegression()` in line 83
```
sklearn/cross_decomposition/_pls.py:107: ConvergenceWarning: Maximum number of iterations reached
  warnings.warn("Maximum number of iterations reached", ConvergenceWarning)
```

> Reference: Liu, S., Zhou, J., Qiu, Y., Chen, J., Zhu, X., & Chen, H. (2022). The FIRST model: Spatiotemporal fusion incorrporting spectral autocorrelation. Remote Sensing of Environment, 279, 113111. https://doi.org/10.1016/j.rse.2022.113111
