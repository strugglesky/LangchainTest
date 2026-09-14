from typing import TypedDict, NotRequired, Required
from typing import cast
from numba.cuda.cudadrv.nvvm import cas_nvvm
from win32comext.mapi.emsabtags import PR_EMS_AB_RAS_PHONE_NUMBER


class PersonInfo(TypedDict):
    name: Required[str]
    age: Required[int]
    address: NotRequired[str]

def print_person(person: PersonInfo):
    print(f'name: {person["name"]}')
    print(f'age: {person["age"]}')
    print(f'address: {person["address"]}')

d1: PersonInfo = {"name": "z3", "age": 18, "address": "beijing"}
print(type(d1))
print_person(d1)