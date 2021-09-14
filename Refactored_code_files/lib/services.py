import json
import numpy
import sqlite3

from lib import utils, models



class AttendanceDatabase:

    def __init__(self):
        print(f"{self.__class__.__name__} : Initating the database connections")
        self.connection = None
        self.database_file_name = "student_attendance.db"
        self.attendance_table = "student_attendance"
        self.signature_table = "student_signature"
        self.connection  = self._get_connnection()
        self.cursor = self.connection.cursor()
        self._create_table()
    
    def _get_connnection(self):
        try:
            if not self.connection:
                return sqlite3.connect(self.database_file_name)
            return self.connection
        except Exception as ex:
            print(f"{self.__class__.__name__}: ERROR: Cannot initate the database connection")
            raise ex

    def _create_table(self):
        try:
            rslt = self.cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.attendance_table} (
                    index_no INT PRIMARY KEY,
                    student_name TEXT NOT NULL,
                    signature BLOB ,
                    attendance_count JSON
                )
                """
            )
            rslt = self.cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.signature_table} (
                    lecture_day INT,
                    index_no INT,
                    signature BLOB,
                    PRIMARY KEY (lecture_day, index_no)
                )
                """
            )
            self.connection.commit()
        except Exception as ex:
            print(f"{self.__class__.__name__}: ERROR: Creating the database")
            self.close_connection()
            raise ex

    def execute_query(self, sql_query, parameters=None) -> sqlite3.Cursor:
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
        self.imageUtils = utils.ImageProcessUtil()
        if xml_students_data:
            self._validate_students_records(xml_students_data)
        last_lecture = self.last_signature_updated_lecture()
        self.lecture_day = 1 if (last_lecture == '' or last_lecture == None) else last_lecture + 1


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

    def create_student_record(self, student: models.Student):
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
            return [ models.Student(*data) for data in data_list]
        return []

    def get_student_record(self, id) -> models.Student:
        fetched_data = self.database.execute_query(
            f"""
            SELECT * FROM {self.database.attendance_table} 
            WHERE index_no = {id}
            """
        ).fetchone()
        if fetched_data:
            return models.Student(*fetched_data)
        return None

    def update_student(self, student:models.Student) -> models.Student:
        updated = self.database.execute_query(
            f"""
            UPDATE {self.database.attendance_table} SET
                    student_name={student.name},
                    signature={student.signature}
            WHERE index_no={student.index}
            """
        )
        if updated:
            return models.Student(*updated)
        return None

    def update_student_attendance(self, student:models.Student, present: bool) -> models.Student:
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
        return None

    def update_signature_if_none(self, student:models.Student, signature_image: numpy.ndarray) -> models.Student:
        if not student.signature.size:
            encoded_image = self.imageUtils.encode_image(signature_image)
            updated = self.database.execute_query(
                f"""
                UPDATE {self.database.attendance_table} SET 
                    signature=(?)
                WHERE index_no=(?)
                """,
                (encoded_image, student.index),
            )
            print(f"updated row count : {updated.rowcount}")
            if updated.rowcount:
                student.signature = signature_image
                return student
        return student

    def remove_student(self, student:models.Student) -> int:
        return self.database.execute_query(
            f"""
            DELETE FROM {self.database.attendance_table} 
            WHERE index_no={student.index}
            """
        ).rowcount


    def last_signature_updated_lecture(self):
        executed = self.database.execute_query(
            f"""
            SELECT MAX(lecture_day) FROM {self.database.signature_table}
            """
        ).fetchone()
        if executed:
            return executed[0]
        return None

    def create_signature_record(self, student_index, signature_image):
        encoded_image = self.imageUtils.encode_image(signature_image)
        created = self.database.execute_query(
                f"""
                INSERT INTO {self.database.signature_table} (lecture_day, index_no, signature) 
                VALUES (?, ?, ?)
                """,
                (self.lecture_day, student_index, encoded_image),
        )
        return created.rowcount if created else None

    def all_signatures_for_student(self, student_index):
        fetched_data = self.database.execute_query(
            f"""
            SELECT * FROM {self.database.signature_table} 
            WHERE index_no = {student_index}
            """
        ).fetchall()
        if fetched_data:
            return [ models.SignatureRecord(*data) for data in fetched_data]
        return []