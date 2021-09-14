import cv2
import numpy as np
import pytesseract

from scipy import ndimage
from skimage import io
from skimage.filters import threshold_otsu 
from skimage.measure import regionprops
from sklearn.cluster import KMeans


from lib import utils

STUDENT_NO = "studentno"
STUDENT_NAME = "studentname"
SIGNATURE =  "signature"
COORDINATES = "coordinates"

class TesseractOcrParser:

    def __init__(self):
        print(f"{self.__class__.__name__}: INFO: Initailizing the Tesseract parser")
        self.config = r"--oem 3 --psm 6"

    def _preprocess_image(self, image):
        gray_scale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh, bwImage = cv2.threshold(gray_scale_image, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(bwImage, cv2.MORPH_CLOSE, kernel)

    def _process_parsed_value(self, parsed_string):
        return "".join([
            ltr.lower() for ltr in parsed_string if ltr.isalnum()
        ])

    def get_string_from_image(self, image):
        image = self._preprocess_image(image)
        parsed_value = pytesseract.image_to_string(
            image, config=self.config,
        )
        # print(f"BEFORE RETURN OCR parsed STR value : {parsed_value}")
        return self._process_parsed_value(parsed_value)

    def get_int_from_image(self, image):
        image = self._preprocess_image(image)
        parsed_value = pytesseract.image_to_string(
            image, config=self.config,
        )
        # print(f"BEFORE RETURN OCR parsed INT value : {parsed_value}")
        return int("".join([
            ltr for ltr in parsed_value if ltr.isnumeric()
        ]))


class AttendanceImageProcessor:

    def __init__(self, image: np.ndarray ):
        self.padding = 40
        self.image = None
        self.image_table_contours = None
        self.original_image = image
        self.expected_column_name = [
            STUDENT_NO,
            SIGNATURE,
        ]

        self.image_height, self.image_weight = image.shape[:2]
        self.image_center_coordinates = (self.image_weight //2, self.image_height//2) 
        self.imageProcessUtil = utils.ImageProcessUtil()
        self.ocrParser = TesseractOcrParser()

        self._preprocess_image()
        self._detect_tables_lines()
        print(f"{self.__class__.__name__ }: INFO: Initailized AttendaneImageProcessor")


    def _preprocess_image(self):
        hough_lines = cv2.HoughLines(
            cv2.Canny(
                self.imageProcessUtil.gray_scale_image(self.original_image),
                60,
                120,
                apertureSize=3,
            ),
            1,
            np.pi/180,
            40,
        )
        rho,theta = hough_lines[0][0]
        self.image = cv2.warpAffine(
            self.original_image, 
            cv2.getRotationMatrix2D(self.image_center_coordinates,np.degrees(theta)-90, 1.0), 
            (self.image_weight, self.image_height), 
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

    def get_contours_under_region(self, region_coordinates, contours):
        region_x_min, region_y_min, region_x_max, region_y_max = region_coordinates
        detected_contours = []
        for contour in contours:
            x, y, width, heigth = cv2.boundingRect(
                contour,
            )
            if (x > region_x_min) and ((x+width) < region_x_max) and (y > region_y_min) and ((y+heigth) < region_y_max):
                detected_contours.append(contour)
        return detected_contours

    def _sort_contours(self,contours, method="left-to-right"):
        print(f"{self.__class__.__name__ }: INFO: Sorting the contours in {method}")
        reverse = False
        i = 0
        if method == "right-to-left" or method == "bottom-to-top":
            reverse = True
        if method == "top-to-bottom" or method == "bottom-to-top":
            i = 1
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
            key=lambda b:b[1][i], reverse=reverse))
        return (contours, boundingBoxes)

    def _is_inside_region(self, contour_coordinates, region_coordinates):
        x_min, y_min, x_max, y_max = contour_coordinates
        region_x_min, region_y_min, region_x_max, region_y_max = region_coordinates
        if (x_min >= region_x_min) and (y_min >= region_y_min) and (x_max <= region_x_max) and (y_max <= region_y_max):
            return True
        return False

    def _detect_tables_lines(self):
        print(f"{self.__class__.__name__}: INFO: Detecting the student table in the image")
        threshold_value = cv2.threshold(
            self.imageProcessUtil.get_black_and_white_image(self.image),
            100,
            255,
            cv2.THRESH_BINARY_INV, + cv2.THRESH_OTSU
        )[1]

        # Detecting the boxes
        kernel_length = np.array(self.image).shape[1] // 80
        # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
        verticle_kernel = self.imageProcessUtil.get_kernel_using_structuring_element(
             (1, kernel_length),
        )
        # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
        hori_kernel = self.imageProcessUtil.get_kernel_using_structuring_element(
            (kernel_length, 1),
        )
        # A kernel of (3 X 3) ones.
        kernel = self.imageProcessUtil.get_kernel_using_structuring_element(
            (3, 3),
        )

        # Morphological operation to detect vertical lines from an image
        verticle_lines_img = cv2.dilate(
            cv2.erode(threshold_value, verticle_kernel, iterations=3), 
            verticle_kernel, 
            iterations=3,
        )
        horizontal_lines_img = cv2.dilate(
            cv2.erode(threshold_value, hori_kernel, iterations=3),
            hori_kernel,
            iterations=3,
        )

        # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
        img_final_bin = cv2.addWeighted(
            verticle_lines_img, 
            0.5, 
            horizontal_lines_img, 
            0.5, 
            0.0,
        )
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
        (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Find contours for image, which will detect all the boxes
        contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort all the contours by top to bottom.
        self.image_table_contours =  self._sort_contours(contours, method="top-to-bottom")[0]

    def get_student_region(self):
        # TODO: Remove large contour of page with area validation
        # contours = list(contours)[1:]
        contours = list(self.image_table_contours)[1:]
        max_area = np.array(
            [ cv2.contourArea(contour) for contour in contours ]
        ).argmax()
        x_min, y_min, region_width, region_heigth = cv2.boundingRect(
            contours[max_area],
        )
        cooridnates = x_min, y_min, x_min + region_width, y_min + region_heigth 
        contours = self.get_contours_under_region(cooridnates, contours)
        return (cooridnates, contours)
        

    def parse_student_details(self, contours, region_coordinates):
        
        region_details = {}
        _, _, region_x_max, region_y_max = region_coordinates
        
        for column in self.expected_column_name:
            for contour in contours[:5]:
                x, y, w, h = cv2.boundingRect(contour)
                cropped_cell = self.image[y+2: y+h-2, x+2: x+w-2]
                ocr_parsed_text = self.ocrParser.get_string_from_image(cropped_cell)
                if set(column).issubset(set(ocr_parsed_text.lower())):
                    print(f"{self.__class__.__name__ }: INFO: Matched column name using OCR - {ocr_parsed_text.lower()}")
                    region_details[column] = {
                        COORDINATES : (x - self.padding, y + h - self.padding, x + w + self.padding, region_y_max + self.padding)
                    }
                    if column == STUDENT_NO:
                        self.student_no_column_area = w*h
                        start_x, start_y, _, start_h = x, y, w, h
                    if column == SIGNATURE:
                        self.signature_column_area = w*h
                    if len(region_details.keys()) == 2: break
        record_by_row = []
        # After removing the header column from contour table
        print(f"{self.__class__.__name__ }: INFO: Detecting student record region")
        student_record_contours = self.get_contours_under_region(
            (start_x-self.padding, (start_y+start_h)-self.padding, region_x_max+self.padding, region_y_max+self.padding),
            contours
        )
        start_x, start_y = start_x, (start_y+start_h) 
        print(f"{self.__class__.__name__ }: INFO: Detecting student no and signature region in the table ..")
        while len(student_record_contours) > 2:
            record_contours = self.get_contours_under_region(
                (start_x-self.padding, start_y-self.padding , region_x_max+self.padding, start_y+start_h+self.padding),
                contours
            )
            details = {}
            for record in record_contours:

                x, y, w, h = cv2.boundingRect(record)
                if self._is_inside_region((x, y, x+w, y+h), (region_details[STUDENT_NO][COORDINATES])) and self.student_no_column_area <= (w*h):
                    details[STUDENT_NO] = self.ocrParser.get_int_from_image(
                        self.image[y: y+h, x: x+w]
                    )
                    start_x, start_y = x, y+h
                elif self._is_inside_region((x, y, x+w, y+h), (region_details[SIGNATURE][COORDINATES])):
                    if self.signature_column_area < (w*h) or (SIGNATURE not in details.keys()):
                        details[SIGNATURE] = [x, y, x+w, y+h]
                    elif (SIGNATURE in details.keys()):
                        sig_w, sig_h = (details[SIGNATURE][2]-details[SIGNATURE][0], details[SIGNATURE][3]-details[SIGNATURE][1])
                        if (sig_w*sig_h) < (w*h):
                            details[SIGNATURE] = [x, y, x+w, y+h]
            record_by_row.append(details)
            student_record_contours = self.get_contours_under_region(
                (start_x-self.padding, start_y-self.padding, region_x_max+self.padding, region_y_max+self.padding),
                contours
            )
        return record_by_row

    def is_attendance_signed(self, cropped_image):
        print(f"{self.__class__.__name__ }: INFO: Validating whether student has signed the sheet")
        values = self.imageProcessUtil.histogram_values_for_pixels(
            self.imageProcessUtil.get_black_and_white_image(cropped_image),
        )
        return values[0]/np.sum(values) > 0.02


class SignatureFeatureExtractor:

    def __init__(self):
        pass

    def _rgb_to_gray(self, img):
        # Converts rgb to grayscale
        print(f"{self.__class__.__name__}: INFO: Converting the RGB to Gray scale image")
        greyimg = np.zeros((img.shape[0], img.shape[1]))
        for row in range(len(img)):
            for col in range(len(img[row])):
                greyimg[row][col] = np.average(img[row][col])
        return greyimg

    def _gray_to_binary(self, img):
        print(f"{self.__class__.__name__}: INFO: Converting the gray scale to binary image")
        # Converts grayscale to binary
        blur_radius = 0.8
        img = ndimage.gaussian_filter(img, blur_radius)
        thres = threshold_otsu(img)
        binimg = img > thres
        binimg = np.logical_not(binimg)
        return binimg

    def _preprocess_image(self, image):
        # gray_scale_image = utils.ImageProcessUtil.gray_scale_image(image)
        binary_image = self._gray_to_binary(
            self._rgb_to_gray(image),
        )
        r, c = np.where(binary_image==1)
        return binary_image[r.min(): r.max(), c.min(): c.max()]


    def _get_ratio_and_centroid_feature(self, image):
        print(f"{self.__class__.__name__}: INFO: Calculating the ratio and centroid features")
        count = 0
        numOfWhites = 0
        a = np.array([0,0])
        for row in range(len(image)):
            for col in range(len(image[0])):
                if image[row][col] == True:
                    count += 1
                    b = np.array([row,col])
                    a = np.add(a,b)
                    numOfWhites += 1
        rowcols = np.array([image.shape[0], image.shape[1]])
        centroid = a/numOfWhites
        centroid = centroid/rowcols
        ratio = count / (image.shape[0] * image.shape[1])
        
        return (
            ratio, 
            (centroid[0], centroid[1]),
        )

    def _get_eccentricity_solidity(self, image):
        print(f"{self.__class__.__name__ }: INFO: Calculating the eccentricity and solidity features")
        properties = regionprops(image.astype("int8"))
        return properties[0].eccentricity, properties[0].solidity

    def _get_skew_kurtosis(self, image):
        print(f"{self.__class__.__name__ }: INFO: Calculating the skewness and kurtosis features")
        image_heigth, image_width = image.shape
        x = range(image_width)
        y = range(image_heigth)
        #calculate projections along the x and y axes
        xp = np.sum(image, axis=0)
        yp = np.sum(image, axis=1)
        #centroid
        cx = np.sum(x*xp)/np.sum(xp)
        cy = np.sum(y*yp)/np.sum(yp)
        #standard deviation
        x2 = (x-cx)**2
        y2 = (y-cy)**2
        sx = np.sqrt(np.sum(x2*xp)/np.sum(image))
        sy = np.sqrt(np.sum(y2*yp)/np.sum(image))
        
        #skewness
        x3 = (x-cx)**3
        y3 = (y-cy)**3
        skewx = np.sum(xp*x3)/(np.sum(image) * sx**3)
        skewy = np.sum(yp*y3)/(np.sum(image) * sy**3)

        #Kurtosis
        x4 = (x-cx)**4
        y4 = (y-cy)**4
        kurtx = np.sum(xp*x4)/(np.sum(image) * sx**4) - 3
        kurty = np.sum(yp*y4)/(np.sum(image) * sy**4) - 3

        return (skewx , skewy), (kurtx, kurty)

    def generate_feature_for_image(self, image):
        print(f"{self.__class__.__name__ }: INFO: Calcaluting features from signatures image")
        _preprocessed_image = self._preprocess_image(image)
        ratio, centroid = self._get_ratio_and_centroid_feature(_preprocessed_image)
        eccentricity, solidity = self._get_eccentricity_solidity(_preprocessed_image)
        skewness, kurtosis = self._get_skew_kurtosis(_preprocessed_image)
        return [
            ratio, centroid[0], centroid[1], 
            eccentricity,
            solidity, 
            skewness[0], skewness[1],
            kurtosis[0], kurtosis[1],
        ]


class SignaturesValidator:

    def __init__(self):
        self.feature_extractor = SignatureFeatureExtractor()
        self.k_means_cluster = KMeans(
            n_clusters=2,
            init='random',
            n_init=10,
            max_iter=100,
        )

    def _retrieve_features_for_image_list(self, image_list):
        images_features_list = []
        for image in image_list:
            images_features_list.append(
                self.feature_extractor.generate_feature_for_image(
                    image
                )
            )
        return images_features_list

    def validate_signatures(self, image_list):
        print(f"{self.__class__.__name__ }: INFO: Validating the signatures from the student")
        features_list = self._retrieve_features_for_image_list(
            image_list
        )
        clustered_signatures = self.k_means_cluster.fit_predict(
            features_list
        )
        similar_signatures = []
        different_signatures = []
        indx = 0
        for cluster in clustered_signatures:
            if cluster == 0:
                similar_signatures.append(indx)
            else:
                different_signatures.append(indx)
            indx += 1
        return similar_signatures, different_signatures
