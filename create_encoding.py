"""
The MarioGAN.jar depends on an encoding that
was built for that project in particular.

In this script, we create ./data/encoding.json,
and use it throughout.
"""
import json
from pathlib import Path

marioGAN_encoding = {
    "X": 0,
    "S": 1,
    "-": 2,
    "?": 3,
    "Q": 4,
    "E": 5,
    "<": 6,
    ">": 7,
    "[": 8,
    "]": 9,
    "o": 10,
    "P": 11    
}

with open(Path("./encoding.json"), "w") as fp:
    json.dump(marioGAN_encoding, fp)
