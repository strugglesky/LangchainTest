from typing import Optional

from pydantic import BaseModel, field_validator, Field


class Person(BaseModel):
    @staticmethod
    def get_default_phones() -> list[str]:
        return ['11234567890']

    name: str
    age: int
    address: Optional[str] = None
    phones: list[str] = Field(default_factory=get_default_phones, validate_default=True)

    @field_validator("phones")
    def valid_phones(cls,v: list[str]) -> list[str]:
        for phone in v:
            if len(phone) != 11 or phone[0] != "1":
                raise ValueError("phone must be 11 digits and start with 1")
        return v



p = Person(name="z3", age=18, address="beijing")

print(p)

