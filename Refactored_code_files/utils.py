import os
import cv2
import pytesseract
import xmltodict
import ast

import numpy as np

from models import Student
from matplotlib import pyplot as plt

class FileHandler:

    def __init__(self):
        pass

    def _read_file(self, file_path):
        try:
            with open(file_path) as xml_file:
                data = xml_file.read()
            return data
        except Exception as ex:
            print(f"{self.__class__.__name__}: ERROR: Failed reading xml file")
            raise ex

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

    def parse_xml_file(self, path) -> list:
        data = self._read_file(path)
        student_list = []
        parsed_data = xmltodict.parse(data)
        for data in list(parsed_data["nsbm"]["students"]["batches"]["batch_15"]["student"]):
            student_list.append(
                Student(
                    data['index'],
                    data['name'],
                    bytes(),
                    []
                )
            )
        return student_list if student_list else None



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


class ImageProcessUtil:

    def __init__(self):
        pass

    def _gray_scale_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise_using_morphology(self, image, method=cv2.MORPH_CLOSE, kernel_size=(3,3)):
        print(f"{self.__class__.__name__}: removing noise using {method}\n")
        kernel = np.ones(kernel_size, np.uint8)
        return cv2.morphologyEx(image, method, kernel)


    def get_black_and_white_image(self, image):
        print(f"{self.__class__.__name__}: processing for B&W image\n")
        ret,bw_image = cv2.threshold(
            self._gray_scale_image(image),
            127,
            255,
            cv2.THRESH_BINARY
        )
        return bw_image

    def histogram_values_for_pixels(self, image, pixels=[0,255]):
        hist, bins = np.histogram(image.flatten(), 256, pixels)
        return [ hist[pixel_value] for pixel_value in pixels ]


    def is_signature_valid(self, cropped_image):
        bw_image = self.get_black_and_white_image(cropped_image)
        values = self.histogram_values_for_pixels(
            self.remove_noise_using_morphology(bw_image)
        )
        return values[0]/np.sum(values) > 0.02


class Visualization:
    def __init__(self,attendence_data):
        self.name = attendence_data[0]
        self.attendence = ast.literal_eval(str(attendence_data[1]))
        self.plot_array = [self.attendence.count(1),self.attendence.count(0)]
        self.legend_labels = ['Present','Absent']
        self.xpos = np.arange(len((self.legend_labels)))
        self.colors = ['tab:blue', 'tab:orange']


    def show_pie_plot(self):
        fig, ax = plt.subplots()
        ax.pie(self.plot_array, labels = self.legend_labels, colors = self.colors , autopct='%.0f%%')
        ax.set_title(self.name)
        plt.show()

    def show_bar_plot(self):
        plt.bar(self.xpos,self.plot_array,color = self.colors)

        plt.xticks(self.xpos,self.legend_labels)
        plt.ylabel("Count")
        plt.title('Student Monitoring')
        plt.legend()
        plt.show()