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

    attendanceDatabase = StudentAttendanceService()

    student_details = attendanceDatabase.get_student_record(arguments.index_no) 
    if student_details is None:
        raise Exception(f"ERROR: No student found with index number {arguments.index_no}")

    imageVisualize = Visualization(
        student_details.attendance,
        student_details.name
    )
    imageVisualize.show_pie_plot()

    # TODO:  visulization 
    # Create Line graph which is better