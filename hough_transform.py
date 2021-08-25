import cv2 as cv
import numpy as np

img = cv.imread('resized_images/8.jpeg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY) 
edges = cv.Canny(gray,60,120,apertureSize = 3)

lines = cv.HoughLines(edges,1,np.pi/180,40)
print(lines.shape)
for rho,theta in lines[0]:
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	# x1 = int(x0 + 100*(-b))
	# y1 = int(y0 + 100*(a))
	# x2 = int(x0 - 100*(-b))
	# y2 = int(y0 - 100*(a))  
	x1 = int(x0 + 10*(-b))
	y1 = int(y0 + 10*(a))
	x2 = int(x0 - 600*(-b))
	y2 = int(y0 - 600*(a))    
	
	
	print(x1,y1,x2,y2,x0,y0)
	cv.line(img,(x1,y1),(x2,y2),(0,0,255),3)
	
	print('theta',theta)
	print(img.shape)
	(h, w) = img.shape[:2]
	center = (w // 2, h // 2)
	M = cv.getRotationMatrix2D(center,np.degrees(theta)-90, 1.0)
	rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_CUBIC, \
	      borderMode=cv.BORDER_REPLICATE)
	cv.imshow("Rotated", rotated)




cv.imshow("HoughLines", img)
cv.moveWindow("HoughLines", 0, 0)
cv.waitKey(0)

