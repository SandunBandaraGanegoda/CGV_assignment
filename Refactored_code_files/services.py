import os
import json
import sqlite3

from models import Student



class AttendanceDatabase:

    def __init__(self):
        print(f"{self.__class__.__name__} : Initating the database connections")
        self.connection = None
        self.database_file_name = "student_attendance.db"
        self.attendance_table = "student_attendance" 
        self.connection  = self._get_connnection()
        self.cursor = self.connection.cursor()
        self._create_table(self.attendance_table)
    
    def _get_connnection(self):
        try:
            if not self.connection:
                return sqlite3.connect(self.database_file_name)
            return self.connection
        except Exception as ex:
            print(f"{self.__class__.__name__}: ERROR: Cannot initate the database connection")
            raise ex

    def _create_table(self, table_name):
        try:
            rslt = self.cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    index_no INT PRIMARY KEY,
                    student_name TEXT NOT NULL,
                    signature BLOB ,
                    attendance_count JSON
                )
                """
            )
            self.connection.commit()
        except Exception as ex:
            print(f"{self.__class__.__name__}: ERROR: Creating the database")
            self.close_connection()
            raise ex

    def execute_query(self, sql_query, parameters=None):
        try:
            if parameters:
                rslt = self.cursor.execute(sql_query, parameters)
            else:   
                rslt =  self.cursor.execute(sql_query)
            self.connection.commit()
            return rslt
        except Exception as ex:
            print(f"SQL ERROR: {ex}\n")
            return None

    def close_connection(self):
        self.connection.close()


class StudentAttendanceService:

    def __init__(self, xml_students_data=None):
        print(f"{self.__class__.__name__}: INFO: Initating attendance services")
        self.database = AttendanceDatabase()
        if xml_students_data:
            self._validate_students_records(xml_students_data)

    def _validate_students_records(self, student_details):
        data = self.get_all_students()
        records = [] if not data else [
            student.index for student in data
        ]
        print(f"{self.__class__.__name__}: INFO: Found records \n {records}")
        for student in student_details:
            if student.index not in records:
                print(f"{self.__class__.__name__}: INFO: Inserting student record {student.index}")
                self.create_student_record(student)

    def create_student_record(self, student: Student):
        created = self.database.execute_query(
                f"""
                INSERT INTO {self.database.attendance_table} VALUES (?, ?, ?, ?)
                """,
                student.to_db_format(),
        )
        return created.rowcount if created else None 

    def get_all_students(self) -> list:
        data_list = self.database.execute_query(
            f"select * from {self.database.attendance_table}"
        ).fetchall()
        if data_list:
            return [ Student(*data) for data in data_list]
        return []

    def get_student_record(self, id) -> Student:
        fetched_data = self.database.execute_query(
            f"""
            SELECT * FROM {self.database.attendance_table} 
            WHERE index_no = {id}
            """
        ).fetchone()
        if fetched_data:
            return Student(*fetched_data)
        return None

    def update_student(self, student:Student) -> Student:
        updated = self.database.execute_query(
            f"""
            UPDATE {self.database.attendance_table} SET
                    student_name={student.name},
                    signature={student.signature}
            WHERE index_no={student.index}
            """
        )
        if updated:
            return Student(*updated)
        return None

    def update_student_attendance(self, student:Student, present: bool) -> Student:
        attendance = student.attendance[:] 
        value = 1 if present else 0
        attendance.append(value)
        updated = self.database.execute_query(
            f"""
            UPDATE {self.database.attendance_table} SET 
                attendance_count='{json.dumps(attendance)}'
            WHERE index_no={student.index}
            """
        )
        if updated.rowcount:
            student.attendance = attendance
            return student
            # return Student(*updated)
        return None

    def remove_student(self, student:Student) -> int:
        return self.database.execute_query(
            f"""
            DELETE FROM {self.database.attendance_table} 
            WHERE index_no={student.index}
            """
        ).rowcount