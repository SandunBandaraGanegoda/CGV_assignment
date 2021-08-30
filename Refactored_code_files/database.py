import os
import sqlite3
from sqlite3.dbapi2 import connect


class StudentAttendanceDatabase:

    def __init__(self):
        self.database_file_name = "student_attendance.db"
        self.table_name = "student_attendance" 
        self.connection  = self._get_connnection()
        self.cursor = self.connection.cursor()
        # TODO: Create the required tables if doesn't exists already
        # COLUMNS : 
        #   INDEX_NO (INT), STUDENT_NAME (TEXT), SIGNATURE_IMAGE(BLOB), ATTENDENCE_COUNT (INT)
    
    def _get_connnection(self):
        if not self.connection:
            return sqlite3.connect(self.database_file_name)
        return self.connection

    def insert_into(self, columns, values):
        pass

    def remove_record(self, id):
        pass

    def select_record(self, id, columns):
        pass