from PIL import Image
import numpy as np

width , height = 512 , 512
x = np.linspace(0 ,255, width)
y = np.linspace(0 ,255, height)
xx , yy = np.meshgrid(x , y)
img = ( xx + yy )/ 2
img = img+np.random.normal(0,30,img.shape)
img = np.clip(img,0,255).astype(np.uint8)

Image.fromarray(img).save("test_input.bmp" , "BMP")
print("test_input.bmp")