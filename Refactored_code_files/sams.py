import os
import cv2
import argparse
import numpy as np

from models import Student
from utils import TesseractOcrParser, FileHandler, ImageProcessUtil
from services import StudentAttendanceService

WARP_DIR_NAME = "warp_images"
BINARY_DIR_NAME ="binary_image"
COLUMN_IDENTIFIED_DIR_NAME = "column_identify"
CROPPED_IMG_DIR_NAME = "cropped_img"


STUDENT_NO = "studentno"
STUDENT_NAME = "studentname"
SIGNATURE =  "signature"

def drawHoughLines(image, lines, size, color):
    size = size if len(lines) > size else len(lines)
    print(f" number of lines : {size}")
    for line in lines[:size]:
        # print(f" Line : {line}")
        for rho,theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center,np.degrees(theta)-90, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
                borderMode=cv2.BORDER_REPLICATE)
        # return rotated
    return image



def preprocess_image(image):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    print("Converting the image to gray scale")
    gray_scale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canney detection
    canney_edges = 	cv2.Canny(gray_scale_img, 60, 120, apertureSize = 3)
    hough_lines = cv2.HoughLines(canney_edges, 1, np.pi/180, 40)
    rho,theta = hough_lines[0][0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
	
    x1 = int(x0 + 10*(-b))
    y1 = int(y0 + 10*(a))
    x2 = int(x0 - 600*(-b))
    y2 = int(y0 - 600*(a))    

    M = cv2.getRotationMatrix2D(center,np.degrees(theta)-90, 1.0)
    aligned_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
		borderMode=cv2.BORDER_REPLICATE)
    return aligned_image


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
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)


def detect_tables_lines(image):
    print("Detecting the table in the image")
    ret, bw_image = cv2.threshold(
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY
    )
    
    threshold_value = cv2.threshold(
        bw_image, 100, 255, cv2.THRESH_BINARY_INV, + cv2.THRESH_OTSU
    )[1]

    # Detecting the boxes
    kernel_length = np.array(image).shape[1] // 80
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
	# A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
	# A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    # Morphological operation to detect vertical lines from an image
    img_temp1 = cv2.erode(threshold_value, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    img_temp2 = cv2.erode(threshold_value, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    return sort_contours(contours, method="top-to-bottom")


def get_contours_under_region(region_coordinates, contours):
    region_x_min, region_y_min, region_x_max, region_y_max = region_coordinates
    detected_contours = []
    for contour in contours:
        x, y, width, heigth = cv2.boundingRect(
            contour,
        )
        if (x > region_x_min) and ((x+width) < region_x_max) and (y > region_y_min) and ((y+heigth) < region_y_max):
            detected_contours.append(contour)
    # if len(detected_contours) == 0:
    #     return []
        # raise Exception("Error no contours detected under the region coordinates")
    return detected_contours

def is_inside_region(contour_coordinates, region_coordinates):
    x_min, y_min, x_max, y_max = contour_coordinates
    region_x_min, region_y_min, region_x_max, region_y_max = region_coordinates
    if (x_min >= region_x_min) and (y_min >= region_y_min) and (x_max <= region_x_max) and (y_max <= region_y_max):
        return True
    return False

def get_student_region(contours):
    # TODO: Remove large contour of page with area validation
    contours = list(contours)[1:]
    max_area = np.array(
        [ cv2.contourArea(contour) for contour in contours ]
    ).argmax()
    x_min, y_min, region_width, region_heigth = cv2.boundingRect(
        contours[max_area],
    )
    cooridnates = x_min, y_min, x_min + region_width, y_min + region_heigth 
    contours = get_contours_under_region(cooridnates, contours)
    return (cooridnates, contours)
    


def parse_student_details(image, contours, region_coordinates, op_path):
    ocrParser = TesseractOcrParser()
    expected_column_name = [
        STUDENT_NO,
        SIGNATURE,
    ]
    padding = 40
    region_details = {}
    region_x_min, region_y_min, region_x_max, region_y_max = region_coordinates
    
    for column in expected_column_name:
        for contour in contours[:5]:
            x, y, w, h = cv2.boundingRect(contour)
            cropped_cell = image[y+2: y+h-2, x+2: x+w-2]
            ocr_parsed_text = ocrParser.get_string_from_image(cropped_cell)
            if set(column).issubset(set(ocr_parsed_text.lower())):
                print(f"Matched OCR : {ocr_parsed_text.lower()}")
                region_details[column] = {
                    "coordinates" : (x - padding, y + h - padding, x + w + padding, region_y_max + padding)
                }
                if column == STUDENT_NO:
                    print(f"Setting studentno has start coordinates: {(x, y + h , w, h)}")
                    start_x, start_y, start_w, start_h = x, y, w, h
                if len(region_details.keys()) == 2: break
    record_by_row = []
    # After removing the header column from contour table
    coord = (start_x-padding, (start_y+start_h)-padding, region_x_max+padding, region_y_max+padding)
    print("Detecting student record region")
    student_record_contours = get_contours_under_region(
        coord,
        contours
    )
    index = 0
    start_x, start_y = start_x, (start_y+start_h) 
    while len(student_record_contours) > 2:
        record_contours = get_contours_under_region(
            (start_x-padding, start_y-padding , region_x_max+padding, start_y+start_h+padding),
            contours
        )

        # img = cv2.rectangle(image, (start_x- padding, start_y-padding), (region_x_max+padding, start_y+start_h+padding), (0, 0, 255), 5)
        # cv2.imwrite(os.path.join(op_path, f"draw_{index}.jpeg"), img)
        details = {}
        print("Detecting student no and signature region in the table ..")
        for record in record_contours:
            x, y, w, h = cv2.boundingRect(record)
            if is_inside_region((x, y, x+w, y+h), (region_details["studentno"]["coordinates"])):
                details["studentno"] = ocrParser.get_int_from_image(
                    image[y: y+h, x: x+w]
                )
                start_x, start_y = x, y+h
                # print("Found cell under student no region")
            if is_inside_region((x, y, x+w, y+h), (region_details["signature"]["coordinates"])):
                # TODO: Validate signature image with the size with average size
                details["signature"] = [x, y, x+w, y+h]
                # print("Found cell under signature region")
        record_by_row.append(details)
        coord = (start_x-padding, start_y-padding, region_x_max+padding, region_y_max+padding)

        student_record_contours = get_contours_under_region(
            coord,
            contours
        )
        index+=1
    return record_by_row


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("image", help="Student signing sheet image")
    argument_parser.add_argument("xml", help="Xml file with student details")
    arguments = argument_parser.parse_args()

    assert os.path.isfile(arguments.image) and os.path.basename(arguments.image).split(".")[1] in ["jpeg", "png"], "Image argument should be image file"
    assert os.path.isfile(arguments.xml) and os.path.basename(arguments.xml).split(".")[1] == "xml", "Image argument should be image file"

    fileHandler = FileHandler()
    student_xml_data = fileHandler.parse_xml_file(arguments.xml)
    
    imageProcessUtil = ImageProcessUtil()

    print(f"Loading the students attendance service...")
    attendanceService = StudentAttendanceService(student_xml_data)

    current_working_dir = os.getcwd()
    output_dir_path = os.path.join(current_working_dir, "output")
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)
    

    original_image = cv2.imread(
        arguments.image
    )

    aligned_image = preprocess_image(original_image)

    contours, boundingBoxes = detect_tables_lines(aligned_image)
    region_cooridnates, student_region_contours = get_student_region(
        contours,
    )

    students_details = parse_student_details(
        aligned_image,
        student_region_contours, # Retrieves the coordinates under the region
        region_cooridnates, 
        output_dir_path
    )
    print(f"Student xml : \n {student_xml_data}\n")

    detected_students = {
        int(student["studentno"]): student["signature"] for student in students_details
    }
    print(f" Detected students : \n{detected_students}")

    for student in attendanceService.get_all_students():
        if student.index in detected_students.keys():
            x_min, y_min, x_max, y_max = detected_students[student.index]
            cropped_image = aligned_image[y_min:y_max, x_min:x_max]
            
            # if imageProcessUtil.is_signature_valid(altered_signature):
            attendanceService.update_student_attendance(
                student, 
                imageProcessUtil.is_signature_valid(cropped_image)
            )
            print(f"Attendence record of {student.name}:  {student.attendance}")

    print("Completed running script...")

