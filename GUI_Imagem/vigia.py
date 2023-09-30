import os
import numpy as np
import cv2
import matplotlib.pylab as plt
from skimage import measure
from skimage.color import label2rgb
from scipy import ndimage as nd



def segementation(img_entrada):  

    img = cv2.imread(img_entrada)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 65, 255, cv2.THRESH_BINARY)

    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    thresh_hsv = cv2.cvtColor(thresh_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(thresh_hsv, (0, 0, 255), (180, 100, 255))
    mask = cv2.medianBlur(mask, 9)

    closed_mask = nd.binary_closing(mask, np.ones((1, 1)))
    label_image = measure.label(closed_mask)

    image_label_overlay = label2rgb(label_image, image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    nome_arquivo = os.path.basename(img_entrada)
    salvar_imagem(nome_arquivo, image_label_overlay)

def salvar_imagem(nome_arquivo, image_label_overlay):
    diretorio_atual = os.path.dirname(__file__)  
    caminho_completo = os.path.join(f"{diretorio_atual}\img\segmentation", nome_arquivo)
    plt.imsave(caminho_completo, image_label_overlay)





