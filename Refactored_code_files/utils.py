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

    def read_image_file_via_cv(self, file_path):
        print(f"{self.__class__.__name__}: INFO: Reading image {file_path}")
        return cv2.imread(file_path)

    def write_image_file_via_cv(self, dir_path, file_name):
        print(f"{self.__class__.__name__}: INFO: Writing image to directory: {dir_path}")
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





class ImageProcessUtil:

    def __init__(self):
        pass

    def gray_scale_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise_using_morphology(self, image, method=cv2.MORPH_CLOSE, kernel_size=(3,3)):
        print(f"{self.__class__.__name__}: removing noise using {method}\n")
        kernel = np.ones(kernel_size, np.uint8)
        return cv2.morphologyEx(image, method, kernel)


    def get_black_and_white_image(self, image):
        print(f"{self.__class__.__name__}: processing for B&W image\n")
        ret,bw_image = cv2.threshold(
            self.gray_scale_image(image),
            127,
            255,
            cv2.THRESH_BINARY
        )
        return bw_image

    def histogram_values_for_pixels(self, image, pixels=[0,255]):
        noise_removed_image = self.remove_noise_using_morphology(image)
        hist, bins = np.histogram(noise_removed_image.flatten(), 256, pixels)
        return [ hist[pixel_value] for pixel_value in pixels ]


    def get_kernel_using_structuring_element(self, size, morph_shape=cv2.MORPH_RECT):
        return cv2.cv2.getStructuringElement(morph_shape, size)


class Visualization:

    def __init__(self, attendence_data, student_name):
        self.name = student_name
        self.attendence = attendence_data
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