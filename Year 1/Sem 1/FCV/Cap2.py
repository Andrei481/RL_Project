# FCV C2 examples
# Dan Pescaru, 2021
import cv2
import numpy as np
import tkinter.filedialog
from matplotlib import pyplot as plt

ex = 1
img = []

def pickColor(event, x, y, flags, param):    
    if event == cv2.EVENT_LBUTTONDOWN:
        pix = imgBGR_new[y, x]
        print(pix, 'at ', x, ', ',y)

root = tkinter.Tk()
img_file = tkinter.filedialog.askopenfilename(initialdir=".", title="Select image file", filetypes=(("Image files", "*.jpg;*.png"), ("all files", "*.*")))
root.destroy()

#Histograms
if(ex==1):
    imgBGR = cv2.imread(img_file)  # read a color image
    imgGray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([imgGray],[0],None,[256],[0,256])
    color = ('b','g','r')
    histrB = cv2.calcHist([imgBGR],[0],None,[256],[0,256])
    histrG = cv2.calcHist([imgBGR],[1],None,[256],[0,256])
    histrR = cv2.calcHist([imgBGR],[2],None,[256],[0,256])

    #plt.subplot(221), plt.imshow(imgBGR)
    plt.subplot(221), plt.imshow(cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB))
    plt.subplot(222), plt.plot(histrR,'r'), plt.plot(histrG,'g'), plt.plot(histrB,'b')
    plt.subplot(223), plt.imshow(imgGray, 'gray')
    plt.subplot(224), plt.plot(hist)
    plt.show()
    exit()

# Contrast and Bightness
if(ex==2):
    img = cv2.imread(img_file)  # read grayscale image
    print(img.shape)
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    contrast = 1.5 #1.5 #0.5 #1.3 #0.5
    brightness = -100 #-100 #-50 #-150
    imgA = img.copy()
    imgA = imgA.astype('float32')
    imgA = (imgA*contrast + brightness).clip(0.0,255.0)
    imgA = imgA.astype('uint8')
    histA = cv2.calcHist([imgA],[0],None,[256],[0,256])
    cv2.imshow('image orig',img)
    cv2.imshow('image axI+b',imgA)
    plt.subplot(121), plt.plot(hist)
    plt.subplot(122), plt.plot(histA)
    plt.show()
    exit()

# Chanel point processings
if(ex==3):
    imgBGR = cv2.imread(img_file)  # read a color image
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", pickColor)
    cv2.imshow('image orig',imgBGR)

    # grayscale
    imgBGR_new = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    imgBGR_new=imgBGR_new.astype('uint8') 
    cv2.putText(imgBGR_new,'RGB/BGR switch',(10,20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,127),1)
    cv2.imshow('image',imgBGR_new)
    cv2.waitKey(0)

    # color changes
    alpha = 1.7
    imgBGR_new = imgBGR.copy()
    imgBGR_new[:, :, 0] = (imgBGR_new[:, :, 0] * alpha).clip(0,255) 
    imgBGR_new=imgBGR_new.astype('uint8') 
    cv2.putText(imgBGR_new,'Blue + 70%',(10,20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,127),1)
    cv2.imshow('image',imgBGR_new)
    cv2.waitKey(0)

    alpha = 1.7
    imgBGR_new = imgBGR.copy()
    imgBGR_new[:, :, 1] = (imgBGR_new[:, :, 1] * alpha).clip(0,255) 
    imgBGR_new=imgBGR_new.astype('uint8') 
    cv2.putText(imgBGR_new,'Green + 70%',(10,20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,127),1)
    cv2.imshow('image',imgBGR_new)
    cv2.waitKey(0)

    alpha = 1.7
    imgBGR_new = imgBGR.copy()
    imgBGR_new[:, :, 2] = (imgBGR_new[:, :, 2] * alpha).clip(0,255) 
    imgBGR_new=imgBGR_new.astype('uint8') 
    cv2.putText(imgBGR_new,'Red + 70%',(10,20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,127),1)
    cv2.imshow('image',imgBGR_new)
    cv2.waitKey(0)

    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    alpha = 1.7
    imgBGR_new  = imgHSV.copy()
    imgBGR_new[:, :, 0] = (imgBGR_new[:, :, 0] * alpha).clip(0,255) 
    imgBGR_new=imgBGR_new.astype('uint8') 
    imgBGR_new = cv2.cvtColor(imgBGR_new, cv2.COLOR_HSV2BGR)
    cv2.putText(imgBGR_new,'H + 70%',(10,20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,127),1)
    cv2.imshow('image',imgBGR_new)
    cv2.waitKey(0)

    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    alpha = 100
    imgBGR_new  = imgHSV.copy()
    imgBGR_new[:, :, 0] = (imgBGR_new[:, :, 0] + alpha).clip(0,255) 
    imgBGR_new=imgBGR_new.astype('uint8') 
    imgBGR_new = cv2.cvtColor(imgBGR_new, cv2.COLOR_HSV2BGR)
    cv2.putText(imgBGR_new,'H + 100',(10,20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,127),1)
    cv2.imshow('image',imgBGR_new)
    cv2.waitKey(0)

    imgHSV = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2HSV)
    alpha = 0.7 #=> go to red
    imgBGR_new  = imgHSV.copy()
    imgBGR_new[:, :, 0] = (imgBGR_new[:, :, 0] * alpha).clip(0,255) 
    imgBGR_new=imgBGR_new.astype('uint8') 
    imgBGR_new = cv2.cvtColor(imgBGR_new, cv2.COLOR_HSV2BGR)
    cv2.putText(imgBGR_new,'H - 30%',(10,20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,127),1)
    cv2.imshow('image',imgBGR_new)
    cv2.waitKey(0)

    alpha = 0.5 
    imgBGR_new  = imgHSV.copy()
    imgBGR_new[:, :, 1] = (imgBGR_new[:, :, 1] * alpha).clip(0,255) 
    imgBGR_new=imgBGR_new.astype('uint8') 
    imgBGR_new = cv2.cvtColor(imgBGR_new, cv2.COLOR_HSV2BGR)
    cv2.putText(imgBGR_new,'S - 50%',(10,20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,127),1)
    cv2.imshow('image',imgBGR_new)
    cv2.waitKey(0)

    alpha = 1.5 
    imgBGR_new  = imgHSV.copy()
    imgBGR_new[:, :, 1] = (imgBGR_new[:, :, 1] * alpha).clip(0,255) 
    imgBGR_new=imgBGR_new.astype('uint8') 
    imgBGR_new = cv2.cvtColor(imgBGR_new, cv2.COLOR_HSV2BGR)
    cv2.putText(imgBGR_new,'S + 50%',(10,20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,127),1)
    cv2.imshow('image',imgBGR_new)
    cv2.waitKey(0)

    alpha = 0.5 
    imgBGR_new  = imgHSV.copy()
    imgBGR_new[:, :, 2] = (imgBGR_new[:, :, 2] * alpha).clip(0,255) 
    imgBGR_new=imgBGR_new.astype('uint8') 
    imgBGR_new = cv2.cvtColor(imgBGR_new, cv2.COLOR_HSV2BGR)
    cv2.putText(imgBGR_new,'V - 50%',(10,20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,127),1)
    cv2.imshow('image',imgBGR_new)
    cv2.waitKey(0)

    alpha = 1.5
    imgBGR_new  = imgHSV.copy()
    imgBGR_new[:, :, 2] = (imgBGR_new[:, :, 2] * alpha).clip(0,255) 
    imgBGR_new=imgBGR_new.astype('uint8') 
    imgBGR_new = cv2.cvtColor(imgBGR_new, cv2.COLOR_HSV2BGR)
    cv2.putText(imgBGR_new,'V + 50%',(10,20), cv2.FONT_HERSHEY_PLAIN, 1,(0,0,127),1)
    cv2.imshow('image',imgBGR_new)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()


# Gamma correction
if(ex==4):
    img = cv2.imread(img_file)  # read grayscale image
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    gamma = 1.2
    imgA = img.copy()
    invGamma = 1.0 / gamma
    # computing lookup table for gamma correction
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    imgA = cv2.LUT(imgA, table)
    histA = cv2.calcHist([imgA],[0],None,[256],[0,256])
    cv2.imshow('image orig',img)
    cv2.imshow('image gamma',imgA)
    plt.subplot(121), plt.plot(hist)
    plt.subplot(122), plt.plot(histA)
    plt.show()
    exit()

    
# filters
if(ex==5):    
    imgBGR = cv2.imread(img_file)  # read a color image
    k = np.array([[-1,-1,-1], [-1, 8,-1], [-1,-1,-1]])
    kG = (1/16) * np.array([[1,2,1], [2, 4, 2], [1, 2, 1]])
    kV = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
    kH = np.array([[0,-1, 0], [0, 1, 0], [0, 0, 0]])    
    kS = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])    
    kC = np.array([[1,-2, 1], [-2, 4,-2], [1,-2, 1]])    
    cv2.imshow('Original', imgBGR)
    img = cv2.filter2D(imgBGR, -1, k)    # -1 result has the same depth as the source
    cv2.imshow('Edges', cv2. bitwise_not(img))
    imgG = cv2.filter2D(imgBGR, -1, kG)                      
    cv2.imshow('Gauss Blur', imgG)
    imgV = cv2.filter2D(imgBGR, -1, kV)                      
    cv2.imshow('Vertical Edges', cv2. bitwise_not(imgV))
    imgH = cv2.filter2D(imgBGR, -1, kH)                      
    cv2.imshow('Horizontal Edges', cv2. bitwise_not(imgH))
    imgS = cv2.filter2D(imgBGR, -1, kS)                      
    cv2.imshow('Sharpening', imgS)
    imgC = cv2.filter2D(imgBGR, -1, kC)                      
    cv2.imshow('Corners', cv2. bitwise_not(imgC))
    cv2.waitKey(0)
    exit()
