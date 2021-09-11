"""
Investigate script
"""
import cv2
import argparse
import numpy as np

from lib.services import StudentAttendanceService



if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("img1", type=str, help="Image 1 ")
    argument_parser.add_argument("img2", type=str, help="Image 2")
    arguments = argument_parser.parse_args()

    # attendanceDatabase = StudentAttendanceService()

    # student_details = attendanceDatabase.get_student_record(arguments.index_no) 
    # if student_details is None:
    #     raise Exception(f"ERROR: No student found with index number {arguments.index_no}")


# 2) Check for similarities between the 2 images

    original = cv2.imread(arguments.img1)
    image_to_compare = cv2.imread(arguments.img2)

    sift = cv2.xfeatures2d.SIFT_create()
    kp_1, desc_1 = sift.detectAndCompute(original, None)
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)


    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_1, desc_2, k=2)


    good_points = []
    ratio = 0.6
    for m, n in matches:
        if m.distance < ratio*n.distance:
            good_points.append(m)
            print(len(good_points))

    
    result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)
    cv2.imshow("result", result)
    cv2.imshow("Original", original)
    cv2.imshow("Duplicate", image_to_compare)
    cv2.waitKey(0)
    cv2.destroyAllWindows()