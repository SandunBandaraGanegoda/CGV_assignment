"""
Investigate script
"""

import cv2
import argparse
import numpy as np

from lib import services, core

from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error



if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("index", type=int, help="Index number of student")
    arguments = argument_parser.parse_args()

    attendanceService = services.StudentAttendanceService()

    student = attendanceService.get_student_record(arguments.index)

    signatures_of_student = attendanceService.all_signatures_for_student(
        arguments.index
    )

    signature_list  = [ sig.signature for sig in signatures_of_student ]
    print(f"LENGHT of student signatures : {len(signature_list)}")
    signature_validator = core.SignaturesValidator()
    similar_signatures, not_similar_signatures = signature_validator.validate_signatures(
        signature_list
    )
    

    print("--------"*6)
    print(f"Signature verification for student\nIndex: {student.index}\nName:{student.name}")
    print("--------"*6)
    print(f"Lectures day with highly similar signatures : {similar_signatures}\n")
    print(f"Different signature detect days {not_similar_signatures}")
    print("--------"*6)




















    # original = cv2.imread(arguments.img1, cv2.IMREAD_GRAYSCALE)
    # template = cv2.imread(arguments.img2, cv2.IMREAD_GRAYSCALE)
    # orig = cv2.resize(original, (433, 77))
    # templ = cv2.resize(template, (433, 77))

    # compare_img(orig, templ)



    # (score, diff) = compare_ssim(orig, templ, full=True)
    # print("SSIM: {}".format(score))

    # thresh = cv2.threshold(diff, 0, 255,
	# cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    # cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	#     cv2.CHAIN_APPROX_SIMPLE)
    # cnts = imutils.grab_contours(cnts)


    # for c in cnts:
    #     # compute the bounding box of the contour and then draw the
    #     # bounding box on both input images to represent where the two
    #     # images differ
    #     (x, y, w, h) = cv2.boundingRect(c)
    #     cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    #     cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # # show the output images
    # cv2.imshow("Original", imageA)
    # cv2.imshow("Modified", imageB)
    # cv2.imshow("Diff", diff)
    # cv2.imshow("Thresh", thresh)
    # cv2.waitKey(0)

    # orb = cv2.ORB_create()
    # original = cv2.Canny(original, 50, 200)
    # template = cv2.Canny(template, 50, 200)

    # # key points and descriptor calculation
    # kp1, desc_1 = orb.detectAndCompute(template, None)
    # kp2, desc_2 = orb.detectAndCompute(original, None)

    # #creating matches
    # matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    # matches_1 = matcher.knnMatch(desc_1, desc_2, 2)
    # print(f"Lenght of matche : {len(matches_1)}")

    # result = cv2.drawMatchesKnn(original, kp1 , template, kp2, matches_1, None)
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    # sift = cv2.SIFT_create()
    # kp_1, desc_1 = sift.detectAndCompute(orig, None)
    # kp_2, desc_2 = sift.detectAndCompute(templ, None)


    # index_params = dict(algorithm=0, trees=5)
    # search_params = dict()
    # flann = cv2.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(desc_1, desc_2, k=2)
    # # Need to draw only good matches, so create a maskte
    # matchesMask = [[0,0] for i in range(len(matches))]
    # # ratio test as per Lowe's paper
    # for i,(m,n) in enumerate(matches):
    #     if m.distance < 0.7*n.distance:
    #         matchesMask[i] = [1, 0]

    # draw_params = dict(matchColor = (0,255,0),
    #                singlePointColor = (255,0,0),
    #                matchesMask = matchesMask,
    #                flags = cv2.DrawMatchesFlags_DEFAULT)
    # result = cv2.drawMatchesKnn(orig ,kp_1, templ ,kp_2,matches,None,**draw_params)
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()