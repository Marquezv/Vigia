```python
!pip install opencv-python
!pip install matplotlib
!pip install scikit-image
!pip install numpy
!pip install stsci.ndimage
!pip install scikit-image

```

    Requirement already satisfied: opencv-python in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (4.8.0.76)
    Requirement already satisfied: numpy>=1.21.2 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from opencv-python) (1.26.0)
    Requirement already satisfied: matplotlib in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (3.8.0)
    Requirement already satisfied: contourpy>=1.0.1 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from matplotlib) (1.1.1)
    Requirement already satisfied: cycler>=0.10 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from matplotlib) (0.11.0)
    Requirement already satisfied: fonttools>=4.22.0 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from matplotlib) (4.42.1)
    Requirement already satisfied: kiwisolver>=1.0.1 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from matplotlib) (1.4.5)
    Requirement already satisfied: numpy<2,>=1.21 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from matplotlib) (1.26.0)
    Requirement already satisfied: packaging>=20.0 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from matplotlib) (23.1)
    Requirement already satisfied: pillow>=6.2.0 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from matplotlib) (10.0.1)
    Requirement already satisfied: pyparsing>=2.3.1 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from matplotlib) (3.1.1)
    Requirement already satisfied: python-dateutil>=2.7 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from matplotlib) (2.8.2)
    Requirement already satisfied: six>=1.5 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from python-dateutil>=2.7->matplotlib) (1.16.0)
    Requirement already satisfied: scikit-image in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (0.21.0)
    Requirement already satisfied: numpy>=1.21.1 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (1.26.0)
    Requirement already satisfied: scipy>=1.8 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (1.11.3)
    Requirement already satisfied: networkx>=2.8 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (3.1)
    Requirement already satisfied: pillow>=9.0.1 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (10.0.1)
    Requirement already satisfied: imageio>=2.27 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (2.31.4)
    Requirement already satisfied: tifffile>=2022.8.12 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (2023.9.26)
    Requirement already satisfied: PyWavelets>=1.1.1 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (1.4.1)
    Requirement already satisfied: packaging>=21 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (23.1)
    Requirement already satisfied: lazy_loader>=0.2 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (0.3)
    Requirement already satisfied: numpy in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (1.26.0)
    Collecting stsci.ndimage
      Using cached stsci.ndimage-0.10.3.tar.gz (99 kB)
      Installing build dependencies: started
      Installing build dependencies: finished with status 'done'
      Getting requirements to build wheel: started
      Getting requirements to build wheel: finished with status 'done'
      Installing backend dependencies: started
      Installing backend dependencies: finished with status 'error'
    

      error: subprocess-exited-with-error
      
      pip subprocess to install backend dependencies did not run successfully.
      exit code: 1
      
      [72 lines of output]
      Collecting stsci.distutils>=0.3.dev
        Using cached stsci.distutils-0.3.7.tar.gz (48 kB)
        Installing build dependencies: started
        Installing build dependencies: finished with status 'done'
        Getting requirements to build wheel: started
        Getting requirements to build wheel: finished with status 'done'
        Installing backend dependencies: started
        Installing backend dependencies: finished with status 'done'
        Preparing metadata (pyproject.toml): started
        Preparing metadata (pyproject.toml): finished with status 'error'
        error: subprocess-exited-with-error
      
        Preparing metadata (pyproject.toml) did not run successfully.
        exit code: 1
      
        [45 lines of output]
        <string>:12: DeprecationWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html
        C:\Users\vmarq\AppData\Local\Temp\pip-build-env-evsqot_m\overlay\Lib\site-packages\setuptools\dist.py:700: SetuptoolsDeprecationWarning: The namespace_packages parameter is deprecated.
        !!
      
                ********************************************************************************
                Please replace its usage with implicit namespaces (PEP 420).
      
                See https://setuptools.pypa.io/en/latest/references/keywords.html#keyword-namespace-packages for details.
                ********************************************************************************
      
        !!
          ep.load()(self, ep.name, value)
        Traceback (most recent call last):
          File "C:\Users\vmarq\AppData\Local\Programs\Python\Python311\Lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 353, in <module>
            main()
          File "C:\Users\vmarq\AppData\Local\Programs\Python\Python311\Lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 335, in main
            json_out['return_val'] = hook(**hook_input['kwargs'])
                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "C:\Users\vmarq\AppData\Local\Programs\Python\Python311\Lib\site-packages\pip\_vendor\pyproject_hooks\_in_process\_in_process.py", line 149, in prepare_metadata_for_build_wheel
            return hook(metadata_directory, config_settings)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "C:\Users\vmarq\AppData\Local\Temp\pip-build-env-evsqot_m\overlay\Lib\site-packages\setuptools\build_meta.py", line 396, in prepare_metadata_for_build_wheel
            self.run_setup()
          File "C:\Users\vmarq\AppData\Local\Temp\pip-build-env-evsqot_m\overlay\Lib\site-packages\setuptools\build_meta.py", line 507, in run_setup
            super(_BuildMetaLegacyBackend, self).run_setup(setup_script=setup_script)
          File "C:\Users\vmarq\AppData\Local\Temp\pip-build-env-evsqot_m\overlay\Lib\site-packages\setuptools\build_meta.py", line 341, in run_setup
            exec(code, locals())
          File "<string>", line 80, in <module>
          File "C:\Users\vmarq\AppData\Local\Temp\pip-build-env-evsqot_m\overlay\Lib\site-packages\setuptools\__init__.py", line 103, in setup
            return distutils.core.setup(**attrs)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "C:\Users\vmarq\AppData\Local\Temp\pip-build-env-evsqot_m\overlay\Lib\site-packages\setuptools\_distutils\core.py", line 147, in setup
            _setup_distribution = dist = klass(attrs)
                                         ^^^^^^^^^^^^
          File "C:\Users\vmarq\AppData\Local\Temp\pip-build-env-evsqot_m\overlay\Lib\site-packages\setuptools\dist.py", line 303, in __init__
            _Distribution.__init__(self, dist_attrs)
          File "C:\Users\vmarq\AppData\Local\Temp\pip-build-env-evsqot_m\overlay\Lib\site-packages\setuptools\_distutils\dist.py", line 283, in __init__
            self.finalize_options()
          File "C:\Users\vmarq\AppData\Local\Temp\pip-build-env-evsqot_m\overlay\Lib\site-packages\setuptools\dist.py", line 680, in finalize_options
            ep(self)
          File "C:\Users\vmarq\AppData\Local\Temp\pip-build-env-evsqot_m\overlay\Lib\site-packages\setuptools\dist.py", line 700, in _finalize_setup_keywords
            ep.load()(self, ep.name, value)
          File "C:\Users\vmarq\AppData\Local\Temp\pip-build-env-evsqot_m\normal\Lib\site-packages\d2to1\core.py", line 30, in d2to1
            from setuptools.dist import _get_unpatched
        ImportError: cannot import name '_get_unpatched' from 'setuptools.dist' (C:\Users\vmarq\AppData\Local\Temp\pip-build-env-evsqot_m\overlay\Lib\site-packages\setuptools\dist.py)
        [end of output]
      
        note: This error originates from a subprocess, and is likely not a problem with pip.
      error: metadata-generation-failed
      
      Encountered error while generating package metadata.
      
      See above for output.
      
      note: This is an issue with the package mentioned above, not pip.
      hint: See above for details.
      [end of output]
      
      note: This error originates from a subprocess, and is likely not a problem with pip.
    error: subprocess-exited-with-error
    
    pip subprocess to install backend dependencies did not run successfully.
    exit code: 1
    
    See above for output.
    
    note: This error originates from a subprocess, and is likely not a problem with pip.
    

    Requirement already satisfied: scikit-image in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (0.21.0)
    Requirement already satisfied: numpy>=1.21.1 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (1.26.0)
    Requirement already satisfied: scipy>=1.8 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (1.11.3)
    Requirement already satisfied: networkx>=2.8 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (3.1)
    Requirement already satisfied: pillow>=9.0.1 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (10.0.1)
    Requirement already satisfied: imageio>=2.27 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (2.31.4)
    Requirement already satisfied: tifffile>=2022.8.12 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (2023.9.26)
    Requirement already satisfied: PyWavelets>=1.1.1 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (1.4.1)
    Requirement already satisfied: packaging>=21 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (23.1)
    Requirement already satisfied: lazy_loader>=0.2 in c:\users\vmarq\appdata\local\programs\python\python311\lib\site-packages (from scikit-image) (0.3)
    


```python
import cv2
import matplotlib.pylab as plt
from skimage.segmentation import felzenszwalb
from skimage.color import rgba2rgb
from skimage import io, measure
import numpy as np
from scipy import ndimage as nd
from skimage.color import label2rgb
```

## Reading Imagens


```python
img = cv2.imread("image.png")
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x23c00126c10>




    
![png](output_3_1.png)
    



```python

```




    <matplotlib.image.AxesImage at 0x1d9bbda1310>




    
![png](output_4_1.png)
    


## Display Images


```python
plt.figure("Original")
img_color = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_color)
```




    <matplotlib.image.AxesImage at 0x23c0744b850>




    
![png](output_6_1.png)
    



```python
plt.figure("Grayscale")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray)
```




    <matplotlib.image.AxesImage at 0x21a2d542c10>




    
![png](output_7_1.png)
    



```python
plt.figure("Grayscale")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap="gray")
```




    <matplotlib.image.AxesImage at 0x23c0095a3d0>




    
![png](output_8_1.png)
    



```python
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
plt.figure("Binary")
plt.imshow(thresh)
```




    <matplotlib.image.AxesImage at 0x23c0095bad0>




    
![png](output_9_1.png)
    


### O algoritimo OTSU separa os pixels reune os pixels com maior variacao, nesse caso, apos a utilizacao do filtro binario a imagen sofre um processo para destacar os grupos (255 ou 0)


```python
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
plt.figure("OTSU")
plt.imshow(thresh, cmap="gray")
```




    <matplotlib.image.AxesImage at 0x23c00044650>




    
![png](output_11_1.png)
    


### Utilizacao do filtro de binario juntamente com a escala de cinza da imagem 


```python
print(ret)
```

    66.0
    


```python
ret, thresh = cv2.threshold(gray, 66, 255, cv2.THRESH_BINARY)
plt.figure("Binary")
plt.imshow(thresh, cmap="gray")
```




    <matplotlib.image.AxesImage at 0x21a330326d0>




    
![png](output_14_1.png)
    


### Inversao do binario


```python
ret, thresh = cv2.threshold(gray, 66, 255, cv2.THRESH_BINARY_INV)
plt.figure("Binary")
plt.imshow(thresh, cmap="gray")
```




    <matplotlib.image.AxesImage at 0x23c00d4e550>




    
![png](output_16_1.png)
    



```python
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_TRIANGLE)
plt.figure("Triangle") 

thresh = np.array(thresh)
thresh[thresh == 255] = 100

plt.imshow(thresh)
print(np.unique(thresh))

```

    [  0 100]
    


    
![png](output_17_1.png)
    



```python
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
laplacian = cv2.Laplacian(img,cv2.CV_64F)
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap = 'gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap = 'gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap = 'gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()


```


    
![png](output_18_0.png)
    


### HSL


```python
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
plt.imshow(img)
```




    <matplotlib.image.AxesImage at 0x21a41286990>




    
![png](output_20_1.png)
    


#### Convert to HSV


```python
ret, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

thresh_hsv = cv2.cvtColor(thresh_bgr, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(thresh_hsv, (0, 0, 250), (180, 70, 255))
plt.imshow(mask)
```




    <matplotlib.image.AxesImage at 0x21a547cc5d0>




    
![png](output_22_1.png)
    



```python
ret, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
thresh_hsv = cv2.cvtColor(thresh_bgr, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(thresh_hsv, (0, 0, 250), (180, 70, 255))

mask = cv2.medianBlur(mask, 5)

thresh = cv2.adaptiveThreshold(mask, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=15, C=0)


plt.imshow(thresh)
```




    <matplotlib.image.AxesImage at 0x21a562d5d90>




    
![png](output_23_1.png)
    



```python
ret, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
thresh_hsv = cv2.cvtColor(thresh_bgr, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(thresh_hsv, (0, 0, 250), (180, 70, 255))

mask = cv2.medianBlur(mask, 5)

thresh = cv2.adaptiveThreshold(mask, 255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=15, C=0)


plt.imshow(thresh)
```




    <matplotlib.image.AxesImage at 0x21a56378c50>




    
![png](output_24_1.png)
    



```python
ret, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

thresh_hsv = cv2.cvtColor(thresh_bgr, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(thresh_hsv, (0, 0, 250), (180, 70, 255))
closed_mask = nd.binary_closing(mask, np.ones((7,7)))
plt.imshow(closed_mask)

```




    <matplotlib.image.AxesImage at 0x21a54818550>




    
![png](output_25_1.png)
    



```python
ret, thresh = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY)
thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

thresh_hsv = cv2.cvtColor(thresh_bgr, cv2.COLOR_BGR2HSV)

mask = cv2.inRange(thresh_hsv, (0, 0, 250), (180, 70, 255))
closed_mask = nd.binary_closing(mask, np.ones((7,7)))
plt.imshow(closed_mask)
```


```python
label_image = measure.label(closed_mask)
plt.imshow(label_image)
```




    <matplotlib.image.AxesImage at 0x21a6bc03450>




    
![png](output_27_1.png)
    



```python
img = io.imread('image.png')

image_label_overlay = label2rgb(label_image, image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.imshow(image_label_overlay)
```




    <matplotlib.image.AxesImage at 0x21a6adc7250>




    
![png](output_28_1.png)
    



```python

```
