import os
import cv2
import pytesseract
import xmltodict

import numpy as np

class FileHandler:

    def __init__(self):
        pass

    def read_image_file(self, file_path):
        print(f"Reading image file {file_path}")
        self.image = cv2.imread(
            file_path,
        )

    def write_image_file(self, dir_path, file_name):
        cv2.imwrite(
            os.path.join(dir_path, file_name),
            self.image,
        )

    def read_xml_file(self, file_path):
        with open(file_path) as xml_file:
            data = xml_file.read()
        return xmltodict.parse(data)


class TesseractOcrParser:

    def __init__(self):
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
        # print(f"OCR parsed value before preprocess : {parsed_value}")
        return self._process_parsed_value(parsed_value)

    def get_int_from_image(self, image):
        image = self._preprocess_image(image)
        parsed_value = pytesseract.image_to_string(
            image, config=self.config,
        )
        return int("".join([
            ltr for ltr in parsed_value if ltr.isnumeric()
        ]))


# class ImageProcessUtil:

#     def __init__(self):
#         pass

#     def 