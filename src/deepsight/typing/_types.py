##
##
##

number = bool | float | int


JSONPrimitive = (
    bool
    | int
    | float
    | str
    | None
    | dict[str, "JSONPrimitive"]
    | list["JSONPrimitive"]
    | tuple["JSONPrimitive"]
)
