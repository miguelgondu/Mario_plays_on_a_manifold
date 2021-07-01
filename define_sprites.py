"""
This script defines the sprites.

NEEDS: ./encoding.json

OUTPUT: a ./sprites.json file containing
the paths to the png images.
"""
import json
from pathlib import Path

with open("./encoding.json") as fp:
    encoding = json.load(fp)


def absolute(path_str):
    return str(Path(path_str).absolute())


sprites = {
    encoding["X"]: absolute("./sprites/stone.png"),
    encoding["S"]: absolute("./sprites/breakable_stone.png"),
    encoding["?"]: absolute("./sprites/question.png"),
    encoding["Q"]: absolute("./sprites/depleted_question.png"),
    encoding["E"]: absolute("./sprites/goomba.png"),
    encoding["<"]: absolute("./sprites/left_pipe_head.png"),
    encoding[">"]: absolute("./sprites/right_pipe_head.png"),
    encoding["["]: absolute("./sprites/left_pipe.png"),
    encoding["]"]: absolute("./sprites/right_pipe.png"),
    encoding["o"]: absolute("./sprites/coin.png"),
}

with open("./sprites.json", "w") as fp:
    json.dump(sprites, fp)
