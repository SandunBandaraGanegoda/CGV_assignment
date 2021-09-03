import os
import argparse

from lib.utils import FileHandler
from lib.services import StudentAttendanceService
from lib.core import AttendanceImageProcessor, STUDENT_NO, SIGNATURE



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

    current_path = os.getcwd()
    op_path = os.path.join(current_path, "outputs")
    if not os.path.exists(op_path):
        os.mkdir(op_path)

    region_cooridnates, student_region_contours = attendanceImageProcessor.get_student_region()

    students_details = attendanceImageProcessor.parse_student_details(
        student_region_contours, # Retrieves the coordinates under the region
        region_cooridnates,
    )
    detected_students = {
        int(student[STUDENT_NO]): student[SIGNATURE] for student in students_details
    }
    for student in attendanceService.get_all_students():
        if student.index in detected_students.keys():
            x_min, y_min, x_max, y_max = detected_students[student.index]
            cropped_image = attendanceImageProcessor.image[y_min:y_max, x_min:x_max]
            attendance_status = attendanceImageProcessor.is_attendance_signed(cropped_image)
            attendanceService.update_student_attendance(
                student, 
                attendance_status,
            )
            if attendance_status:
                attendanceService.update_signature_if_none(
                    student, cropped_image,
                )
            status = "Present" if attendance_status else "Absent" 
            print(f"Attendence record of {student.name}:  {status}")
    print("Completed recording attendance...")
