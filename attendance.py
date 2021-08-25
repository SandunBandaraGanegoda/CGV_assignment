import cv2
# handle directories
import os
# read all images in a folder
from PIL import Image
import glob



# Create folders
binary_img="binary_images"
warp_img = "warp_images"
resized_img = "resized_images"
signing_sheets = "signing_sheets/"

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


path_resized_imges = os.path.join(parent_dir,resized_img)
try:
    os.makedirs(path_resized_imges, exist_ok = True)
    print("Directory '%s' created Successfully" % path_resized_imges)
except OSError as error:
    print("Directory '%s' can not be created" % path_resized_imges)

image_list = []
path_sign_sheets = os.path.join(parent_dir,signing_sheets)
data = os.path.join(path_sign_sheets,"*.jpeg")

for filename in glob.glob(data): #assuming jpeg
    im=cv2.imread(filename)
    image_list.append(im)
    print(filename)
    # print(image_list)

edge_img_list=[]
for i in range(len(image_list)):
    edge_img = cv2.Canny(image_list[i],100,200)
    # cv2.imwrite('assets/EBL.jpg', edge_img)
    edge_img_list.append(edge_img)

resized_img_list=[]
for i in range(len(image_list)):
    resize_img = cv2.resize(image_list[i], (0,0), fx=0.25, fy=0.25)
    # cv2.imwrite('assets/RBL.jpg', newImg)
    resized_img_list.append(resize_img)


for i in range (len(resized_img_list)):
    cv2.imwrite(path_resized_imges+"/"+str(i+1)+".jpeg",resized_img_list[i])

print("working")
    

