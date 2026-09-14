from pydantic import BaseModel, Field, ValidationError
from typing import Annotated


Age = Annotated[int, Field(ge=0, le=150, description="年龄，范围0-150")]

class Person(BaseModel):
    name: str
    age: Age

try:
    p = Person(name="asd", age=1232)
    print(p)
except ValidationError as e:
    print(e)
