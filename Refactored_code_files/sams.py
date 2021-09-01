import os
import argparse

from utils import FileHandler
from services import StudentAttendanceService
from core import AttendanceImageProcessor, STUDENT_NO, SIGNATURE



if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("image", help="Student signing sheet image")
    argument_parser.add_argument("xml", help="Xml file with student details")
    arguments = argument_parser.parse_args()

    assert os.path.isfile(arguments.image) and os.path.basename(arguments.image).split(".")[1] in ["jpeg", "png"], "Image argument should be image file"
    assert os.path.isfile(arguments.xml) and os.path.basename(arguments.xml).split(".")[1] == "xml", "Image argument should be image file"

    fileHandler = FileHandler()
    student_xml_data = fileHandler.parse_xml_file(arguments.xml)
    
    print(f"Loading the students attendance service...")
    attendanceService = StudentAttendanceService(student_xml_data)

    attendanceImageProcessor= AttendanceImageProcessor(
        fileHandler.read_image_file_via_cv(arguments.image)
    )

    region_cooridnates, student_region_contours = attendanceImageProcessor.get_student_region()

    students_details = attendanceImageProcessor.parse_student_details(
        student_region_contours, # Retrieves the coordinates under the region
        region_cooridnates,
    )

    print(f"Student xml : \n {student_xml_data}\n")

    detected_students = {
        int(student[STUDENT_NO]): student[SIGNATURE] for student in students_details
    }
    print(f" Detected students : \n{detected_students}")

    for student in attendanceService.get_all_students():
        if student.index in detected_students.keys():
            x_min, y_min, x_max, y_max = detected_students[student.index]
            cropped_image = attendanceImageProcessor.image[y_min:y_max, x_min:x_max]
            
            attendanceService.update_student_attendance(
                student, 
                attendanceImageProcessor.is_signature_valid(cropped_image)
            )
            print(f"Attendence record of {student.name}:  {student.attendance}")

    print("Completed running script...")

