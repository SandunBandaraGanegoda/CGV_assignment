import json

class Student:

    def __init__(self, index_number: int, student_name: str, student_signature: bytes, attendance):
        self.name = student_name
        self._index = int(index_number) if not isinstance(index_number, int) else index_number
        self.signature =  student_signature
        self._attendance = json.loads(attendance) if isinstance(attendance, str) else attendance

    def __str__(self) -> str:
        return f"{self.__class__.__name__}: {self.index}"

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index_no):
        self._index = int(index_no) if isinstance(index_no, str) else index_no

    @property
    def attendance(self):
        return self._attendance

    @attendance.setter
    def attendance(self, attendance):
        self._attendance = json.loads(attendance) if not isinstance(attendance, list) else attendance

    def to_db_format(self) -> tuple:
        return (
            self.index,
            self.name,
            self.signature,
            json.dumps(self.attendance),
        )
