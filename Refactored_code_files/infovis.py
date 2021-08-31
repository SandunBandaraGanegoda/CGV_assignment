"""
Visualize script to summary the attendance of student 
by using the provided student index number.
"""
import os
import argparse
import numpy as np

from database import StudentAttendanceDatabase


if __name__ == "__main__":

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("index_no", type=int, help="Index number of student")
    arguments = argument_parser.parse_args()

    attendanceDatabase = StudentAttendanceDatabase()

    attendance_data = attendanceDatabase.get_records(
        'student_name,attendance_count', arguments.index_no
    ).fetchone()
    # Return np array [STUDENT_NAME, ATTENDANCE_COUNT]
    attendance_data = np.array(attendance_data)
    
    # TODO:  visulization 
    # Create Line graph which is better