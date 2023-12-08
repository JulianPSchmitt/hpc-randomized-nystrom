import pathlib
from os.path import split

filefolder = pathlib.Path(__file__).parent.resolve()
_FOLDER = split(filefolder)[0]
