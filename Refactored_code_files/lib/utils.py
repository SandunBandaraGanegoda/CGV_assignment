import cv2
import xmltodict
import numpy as np

from lib import models
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

    def write_image_file_via_cv(self,image, file_path):
        print(f"{self.__class__.__name__}: INFO: Writing image to directory: {file_path}")
        cv2.imwrite(
            file_path,
            image,
        )

    def parse_xml_file(self, path) -> list:
        data = self._read_file(path)
        student_list = []
        parsed_data = xmltodict.parse(data)
        for data in list(parsed_data["nsbm"]["students"]["batches"]["batch_15"]["student"]):
            student_list.append(
                models.Student(
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

    def empty_array(self):
        return np.array([])

    def encode_image(self, image):
        if isinstance(image, np.ndarray) and len(image) != 0:
            return np.array(cv2.imencode('.jpg', image)[1]).tobytes()
        return bytes()

    def decode_image(self, image_bytes: bytes):
        if isinstance(image_bytes, bytes) and len(image_bytes) != 0:
            return  cv2.imdecode(
                np.asarray(bytearray(image_bytes ), dtype="uint8"),
                cv2.IMREAD_COLOR
            )
        return self.empty_array()

    def gray_scale_image(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def remove_noise_using_morphology(self, image, method=cv2.MORPH_CLOSE, kernel_size=(3,3)):
        print(f"{self.__class__.__name__}: removing noise using Morph \n")
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

    def __init__(self, student: models.Student, attendance_details):
        self.student = student
        self.students_attendance_details = attendance_details
        self.figure = plt.figure()
        self.figure.suptitle(
            f'Details of student attendance : {self.student.index}\n {self.student.name}',
            fontsize=14, 
            fontweight='bold',
            color="#76726F"
        )

    def _calculate_total_attendace(self):
        attendance_list = self.students_attendance_details
        for att in attendance_list:
            if len(att) < len(self.student.attendance):
                for x in range(len(self.student.attendance) - len(att)):
                    att.append(0)
            if len(att) > len(self.student.attendance):
                for x in range(len(att) - len(self.student.attendance)):
                    self.student.attendance.append(0)
        attendance_list.append(self.student.attendance)

        np_array = np.array(attendance_list)
        return np.sum(np_array, axis=0)

    def generate_pie_plot(self):
        pie_plot = self.figure.add_subplot(2, 2, 4)
        pie_plot.set_title("Overall Student attendance", color="#0066CC" )
        pie_plot.pie(
            [self.student.attendance.count(1), self.student.attendance.count(0)],
            labels=['Present','Absent'],
            colors=['tab:blue', 'tab:orange'],
            autopct='%.0f%%',
            radius=1,
        )

    def generate_line_plot(self):
        line_plot = self.figure.add_subplot(2, 2, 2)
        line_plot.set_title("Attendance by days", color="#0066CC")
        line_plot.plot(
            [x for x in range(1, len(self.student.attendance)+1)],
            self.student.attendance, 
            linewidth = 3,
            marker='o',
            markerfacecolor='orange',
            markersize=12,
        )
        line_plot.set_xlabel("Attendance counted days")
        line_plot.set_yticklabels(["Absent", "Present"], color = "#25517D" )
        line_plot.set_yticks([0, 1])
        line_plot.set_xlim(0, len(self.student.attendance)+1, 1)
        line_plot.set_ylim(-1, 2, 1)

    def generate_bar_plot(self):
        bar_plot = self.figure.add_subplot(1, 2, 1)
        bar_plot.set_title("Total attendance for the lecture", color="#0066CC")
        attendance_for_class = self._calculate_total_attendace()
        bar_plot.set_xlabel("Lecture days")
        bar_plot.set_ylabel("Total attendance count")
        bar_plot.bar(
            [x for x in range(1, len(self.student.attendance) +1)], 
            attendance_for_class,
            width=0.3,
        )


    def show_graph(self):
        self.generate_line_plot()
        self.generate_pie_plot()
        self.generate_bar_plot()
        plt.tight_layout()
        plt.show()
    