"""Board json utilities

FYI see https://linusakesson.net/games/autosokoban/ for where I got some the puzzles
"""

import os
import json


def dehydrate(x):
    return [list(t) for t in list(x)]


def hydrate(x):
    return set([tuple(t) for t in x])


def save_dict_to_json(d, filename):
    assert set(d.keys()) == {"airs", "boxes", "goals", "player"}

    data = d.copy()
    data["airs"] = dehydrate(data["airs"])
    data["boxes"] = dehydrate(data["boxes"])
    data["goals"] = dehydrate(data["goals"])
    data["player"] = list(data["player"])

    print(data)

    try:
        os.remove(filename)
    except OSError:
        pass

    with open(filename, "w") as f:
        json.dump(data, f)


def load_json_to_dict(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    assert set(data.keys()) == {"airs", "boxes", "goals", "player"}

    d = data.copy()
    d["airs"] = hydrate(d["airs"])
    d["boxes"] = hydrate(d["boxes"])
    d["goals"] = hydrate(d["goals"])
    d["player"] = tuple(d["player"])

    return d
