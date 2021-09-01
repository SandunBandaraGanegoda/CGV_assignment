"""
Visualize script to summary the attendance of student 
by using the provided student index number.
"""
import os
import argparse
import numpy as np

from services import StudentAttendanceService
from utils import Visualization


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("index_no", type=int, help="Index number of student")
    arguments = argument_parser.parse_args()

    attendanceDatabase = StudentAttendanceService()

    student_details = attendanceDatabase.get_student_record(arguments.index_no) 
    

    # Return np array [STUDENT_NAME, ATTENDANCE_COUNT]
    print(f"DATA : { np.array(student_details.attendance)}")
    imageVisualize = Visualization(
        np.array(student_details.attendance)
    )
    imageVisualize.show_pie_plot()

    # TODO:  visulization 
    # Create Line graph which is better