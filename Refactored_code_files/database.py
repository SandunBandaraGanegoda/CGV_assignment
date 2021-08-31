import os
import sqlite3
from sqlite3.dbapi2 import connect


class StudentAttendanceDatabase:

    def __init__(self):
        self.connection = None
        self.database_file_name = "student_attendance.db"
        self.table_name = "student_attendance" 
        self.connection  = self._get_connnection()
        self.cursor = self.connection.cursor()
        self._create_table(self.table_name)
    
    def _create_table(self, table_name):
        try:
            rslt = self.cursor.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    index_no INT PRIMARY KEY,
                    student_name TEXT NOT NULL,
                    signature BLOB ,
                    attendance_count INT
                )
                """
            )
            self.connection.commit()
        except Exception as ex:
            print("SQL ERROR : Creating the database")
            raise ex

    def _execute_query(self, sql_query, parameters=None):
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

    def _get_connnection(self):
        try:
            if not self.connection:
                return sqlite3.connect(self.database_file_name)
            return self.connection
        
        except Exception as ex:
            print("SQL ERROR : Cannot initate the database connection")
            raise ex

    def get_all_records(self):
        return self._execute_query(
            f"select * from {self.table_name}"
        )

    def get_records(self, columns, id):
        query = f"select {columns} from {self.table_name} where index_no = {id}"
        return self._execute_query(query)
        

    def insert_into(self, values):
        return self._execute_query(
            f"insert into {self.table_name} (index_no, student_name, signature) values (?, ?, ?)",
            values,
        )

    def update_record(self, column, id, value):
        return self._execute_query(
            f"update {self.table_name} set {column}={value} where index_no = {id}",
        )

    def remove_record(self, id):
        return self._execute_query(
            f"DELETE FROM {self.table_name} WHERE index_no = ?",
            id,
        )