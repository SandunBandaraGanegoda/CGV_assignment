"""
Visualize script to summary the attendance of student 
by using the provided student index number.
"""
import argparse

from lib.services import StudentAttendanceService
from lib.utils import Visualization


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("index_no", type=int, help="Index number of student")
    arguments = argument_parser.parse_args()

    attendanceService = StudentAttendanceService()

    student = attendanceService.get_student_record(arguments.index_no) 
    all_students_details = attendanceService.get_all_students()
    if student is None:
        raise Exception(f"ERROR: No student found with index number {arguments.index_no}")

    imageVisualize = Visualization(
        student,
        [stud.attendance for stud in all_students_details if stud.index != student.index],
    )

    imageVisualize.show_graph()