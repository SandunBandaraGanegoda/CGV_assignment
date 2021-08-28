
#  Test commit
import cv2 as cv
import numpy as np
import os
import glob

def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

warp_img = "warp_images"
binary_img="binary_images"
signing_sheets = "signing_sheets"
# signing_sheets = "signing_sheet_1"
column_identify = "column_identify"
cropped_img = "cropped_img"


parent_dir = os.getcwd()
path_binary_img = os.path.join(parent_dir,binary_img)
try:
    os.makedirs(path_binary_img, exist_ok = True)
    print("Directory '%s' created Successfully" % path_binary_img)
except OSError as error:
    print("Directory '%s' can not be created" % path_binary_img)


path_warp_img = os.path.join(parent_dir,warp_img)
try:
    os.makedirs(path_warp_img, exist_ok = True)
    print("Directory '%s'  created Successfully" %path_warp_img)
except OSError as error:
    print("Directory '%s'  can not be created" %path_warp_img)


path_column_identify_img = os.path.join(parent_dir,column_identify)
try:
    os.makedirs(path_column_identify_img, exist_ok = True)
    print("Directory '%s' created Successfully" % path_column_identify_img)
except OSError as error:
    print("Directory '%s' can not be created" % path_column_identify_img)

cropped_dir_path = os.path.join(parent_dir,cropped_img)
try:
    os.makedirs(cropped_dir_path, exist_ok = True)
    print("Directory '%s' created Successfully" % cropped_dir_path)
except OSError as error:
    print("Directory '%s' can not be created" % cropped_dir_path)



img_list={}
path_sign_sheets = os.path.join(parent_dir,signing_sheets)
data = os.path.join(path_sign_sheets,"*.jpeg")


for filename in glob.glob(data): #assuming jpeg
	im=cv.imread(filename)
	i_name = os.path.basename(filename)
	img_list[i_name] = im
	



warp_img_list={}
binary_img_list={}




# for i in range(len(img_list)):
for img_name in img_list.keys():
	load_img = img_list[img_name]
	gray = cv.cvtColor(load_img,cv.COLOR_BGR2GRAY) 
	edges = cv.Canny(gray,60,120,apertureSize = 3)
	lines = cv.HoughLines(edges,1,np.pi/180,40)
	print(lines.shape)


	# for rho,theta in lines[0]:
	rho,theta = lines[0][0]
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	
	x1 = int(x0 + 10*(-b))
	y1 = int(y0 + 10*(a))
	x2 = int(x0 - 600*(-b))
	y2 = int(y0 - 600*(a))    
	
	
	print(x1,y1,x2,y2,x0,y0)
	# cv.line(img_list[i],(x1,y1),(x2,y2),(0,0,255),3)
	
	print('theta',theta)
	# print(img_list[i].shape)
	(h, w) = img_list[img_name].shape[:2]
	center = (w // 2, h // 2)
	M = cv.getRotationMatrix2D(center,np.degrees(theta)-90, 1.0)
	rotated = cv.warpAffine(img_list[img_name], M, (w, h), flags=cv.INTER_CUBIC, \
		borderMode=cv.BORDER_REPLICATE)
	
	warp_img_list[img_name] = rotated
	
	# otsu
	# median = cv.medianBlur(cv.cvtColor(rotated,cv.COLOR_BGR2GRAY),5)
	ret,bw_img = cv.threshold(cv.cvtColor(rotated,cv.COLOR_BGR2GRAY),128,255,cv.THRESH_BINARY)
	thresh = cv.threshold(bw_img, 128, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1] 

	binary_img_list[img_name] =thresh

	cv.imwrite(path_warp_img+"/"+img_name,rotated)
	print(img_name)
	cv.imwrite(path_binary_img+"/"+img_name,thresh)


	# Boxes Drawing
	# Defining a kernel length
	kernel_length = np.array(rotated).shape[1]//80

	# A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
	verticle_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, kernel_length))

	# A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
	hori_kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_length, 1))

	# A kernel of (3 X 3) ones.
	kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

	# Morphological operation to detect vertical lines from an image
	img_temp1 = cv.erode(thresh, verticle_kernel, iterations=3)
	verticle_lines_img = cv.dilate(img_temp1, verticle_kernel, iterations=3)
	cv.imwrite(path_column_identify_img+"/"+img_name,verticle_lines_img)
	# Morphological operation to detect horizontal lines from an image
	img_temp2 = cv.erode(thresh, hori_kernel, iterations=3)
	horizontal_lines_img = cv.dilate(img_temp2, hori_kernel, iterations=3)
	cv.imwrite(path_column_identify_img+"/"+img_name,horizontal_lines_img)

	# Weighting parameters, this will decide the quantity of an image to be added to make a new image.
	alpha = 0.5
	beta = 1.0 - alpha
	# This function helps to add two image with specific weight parameter to get a third image as summation of two image.
	img_final_bin = cv.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
	img_final_bin = cv.erode(~img_final_bin, kernel, iterations=2)
	(thresh, img_final_bin) = cv.threshold(img_final_bin, 128,255, cv.THRESH_BINARY | cv.THRESH_OTSU)
	cv.imwrite(path_column_identify_img+"/"+img_name,img_final_bin)

	# Find contours for image, which will detect all the boxes
	contours, hierarchy = cv.findContours(img_final_bin, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
	# Sort all the contours by top to bottom.
	(contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

	#b box locations 
	idx = 0
	for c in contours:
		# Returns the location and width,height for every contour
		x, y, w, h = cv.boundingRect(c)
		if (w > 80 and h > 20) and w > 3*h:
			idx += 1
			new_img = rotated[y:y+h, x:x+w]
			cv.imwrite(cropped_dir_path+"/"+str(idx)+img_name, new_img)
# If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
		else:
			idx += 1
			new_img = rotated[y:y+h, x:x+w]
			cv.imwrite(cropped_dir_path+"/"+str(idx)+"_"+img_name, new_img)
