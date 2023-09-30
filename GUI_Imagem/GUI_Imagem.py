from asyncio.windows_events import NULL
from math import exp
from multiprocessing import connection
import PySimpleGUI as sg
import ctypes
import sys
import os
import cv2
import matplotlib.pylab as plt
from skimage.color import label2rgb
from skimage.color import rgba2rgb
import matplotlib.pylab as plt
from PIL import Image, ImageTk
import vigia as vg
ctypes.windll.user32.ShowWindow( ctypes.windll.kernel32.GetConsoleWindow(), 0 )
from configparser import ConfigParser


sg.theme('DarkPurple7')

janelamenu = True

verifica_pasta = os.path.exists('log')
if verifica_pasta == False:
    os.mkdir('log')

while janelamenu:

    config_object = ConfigParser()
    config_object.read("config.ini")
    userinfo = config_object["USERINFO"]
    caminho_imagem_original = (userinfo["caminho_imagem_original"])
    caminho_salvar = (userinfo["caminho_salvar"])

    layout = [
    [sg.Text('========================================================================\n                                                             GUI - IMAGEM\n========================================================================')],

    [sg.Text("Selecione a imagem original: ",justification='l'), sg.Input(''+caminho_imagem_original,key="-ORG-", justification='l'), sg.FileBrowse(),],
    [sg.Button('Gerar',size=(20,5),pad=(50,5)), sg.Button('Cancelar',size=(20,5),pad=(70,5))],
    [sg.Image(key="-IMAGE-")]
    ]

    windowconf = sg.Window('CONFIGURAR - APONTAMENTO', layout, finalize=True, disable_close=False, element_justification='r')
    event, values = windowconf.read()

    caminho_imagem_original = (values["-ORG-"])

    if event == 'Gerar':
        event, values = windowconf.read()

        config_object = ConfigParser()
        config_object.read("config.ini")

        userinfo = config_object["USERINFO"]

        userinfo["caminho_imagem_original"] = caminho_imagem_original

        with open('config.ini', 'w') as conf:
            config_object.write(conf)
        try:
            sg.popup("Imagem gerada!!\n Veja em GUI_Imagem\img\segmentation")
            vg.segementation(caminho_imagem_original)
            continue
        except Exception as erro:
            sg.popup("Erro ao gerar imagem!!!\n",erro)
    if event == 'Cancelar': 
        janelamenu = False
        break
    windowconf.Close()