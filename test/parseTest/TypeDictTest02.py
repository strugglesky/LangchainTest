from typing import TypedDict, Annotated

from pydantic import Field

Age = Annotated[int, Field(ge=0, le=150, description="年龄，范围0-150")]

class Person(TypedDict):
    name: str
    age: int
    age2: Age  # 本质还是 int，元数据 "年龄，范围0-150" 不参与运行时校验

p = Person(name="z3", age=111, age2=188)
print(p)