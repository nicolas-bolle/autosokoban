"""Basic app for editing board jsons
Using pattern-matchin callbacks to handle the variable number of input cells
"""

import numpy as np

from dash import Dash, dcc, html, Input, Output, State, MATCH, ALL, Patch, callback, ctx
import dash_daq as daq

from utilities import save_dict_to_json


app = Dash()


def convert_to_dict(arr_states):
    airs = set()
    boxes = set()
    goals = set()
    player = (0, 0)

    for i in range(len(arr_states)):
        for j in range(len(arr_states[0])):
            val = arr_states[i][j]
            if val != "wall":
                airs.add((i, j))
            if "box" in val:
                boxes.add((i, j))
            if "goal" in val:
                goals.add((i, j))
            if "player" in val:
                player = (i, j)

    return {"airs": airs, "boxes": boxes, "goals": goals, "player": player}


PRINTING_SYMBOLS = {
    "wall": "█",
    "air": " ",
    "box": "□",
    "player": "○",
    "goal": "·",
    "goal and box": "■",
    "goal and player": "●",
}

IMAGES = {
    "wall": html.Img(src="assets/wall.jpg"),
    "air": html.Img(src="assets/air.jpg"),
    "box": html.Img(src="assets/box.jpg"),
    "player": html.Img(src="assets/player.jpg"),
    "goal": html.Img(src="assets/goal.jpg"),
    "goal and box": html.Img(src="assets/goal_and_box.jpg"),
    "goal and player": html.Img(src="assets/goal_and_player.jpg"),
}

N_COLS = 10
N_ROWS = 8

STYLE = {
    "background-color": "white",
    "color": "white",
    "textAlign": "center",
    "margin-left": "5px",
    "margin-top": "0px",
    "width": "65px",
    "height": "60px",
}

STYLE_BUTTON = {
    "textAlign": "center",
    "margin-left": "15px",
    "margin-top": "15px",
    "width": "100px",
    "height": "50px",
}

DEFAULT_STATE = "air"


def make_cell(i, j):
    """Make the (i, j) cell"""
    return html.Button(
        children=IMAGES[DEFAULT_STATE],
        id={"type": "cell", "index": f"{i} {j}"},
        n_clicks=0,
        style=STYLE,
    )


def make_row(m, j):
    """Make the jth row, with m elements"""
    return html.Div(
        [make_cell(i, j) for i in range(m)], id={"type": "row", "index": str(j)}
    )


def make_grid(m, n):
    """Make an m x n grid
    "Reversed" row order so the y index decreases when moving down the page
    """
    return html.Div(list(reversed([make_row(m, j) for j in range(n)])), id="grid")


def make_layout(n_cols, n_rows):
    return [
        html.Button(
            "Save",
            id="save",
            style=STYLE_BUTTON,
        ),
        daq.NumericInput(
            label="Columns",
            labelPosition="bottom",
            value=n_cols,
            id="n_cols",
            min=1,
            max=100,
        ),
        daq.NumericInput(
            label="Rows",
            labelPosition="bottom",
            value=n_rows,
            id="n_rows",
            min=1,
            max=100,
        ),
        make_grid(n_cols, n_rows),
        html.Button(
            "Air",
            id={"type": "state_selector", "index": "air"},
            style=STYLE_BUTTON,
        ),
        html.Button(
            "Wall",
            id={"type": "state_selector", "index": "wall"},
            style=STYLE_BUTTON,
        ),
        html.Button(
            "Box",
            id={"type": "state_selector", "index": "box"},
            style=STYLE_BUTTON,
        ),
        html.Button(
            "Player",
            id={"type": "state_selector", "index": "player"},
            style=STYLE_BUTTON,
        ),
        html.Button(
            "Goal",
            id={"type": "state_selector", "index": "goal"},
            style=STYLE_BUTTON,
        ),
        html.Button(
            "Goal + box",
            id={"type": "state_selector", "index": "goal and box"},
            style=STYLE_BUTTON,
        ),
        html.Button(
            "Goal + player",
            id={"type": "state_selector", "index": "goal and player"},
            style=STYLE_BUTTON,
        ),
        dcc.Store(
            "arr_states",
            # indexed [i][j]
            data=[[DEFAULT_STATE for _ in range(n_rows)] for _ in range(n_cols)],
        ),
        dcc.Store(
            "state_selected",
            data=DEFAULT_STATE,
        ),
    ]


app.layout = html.Div(make_layout(N_COLS, N_ROWS), id="page")


@callback(
    Output("state_selected", "data"),
    Input({"type": "state_selector", "index": ALL}, "n_clicks"),
    State("state_selected", "data"),
)
def click_state(_, state_selected):
    """Updated the selected state"""
    idx = ctx.triggered_id
    if idx is None:
        return state_selected
    return idx["index"]


@callback(
    Output("arr_states", "data"),
    Input({"type": "cell", "index": ALL}, "n_clicks"),
    State("arr_states", "data"),
    State("state_selected", "data"),
)
def cell_click(_, arr_states, state_selected):
    """Updates the selected state of a single cell"""
    # index the cells clicked
    idx = ctx.triggered_id
    if idx is None:
        return arr_states
    i, j = idx["index"].split(" ")
    i, j = int(i), int(j)

    # update states
    arr_states[i][j] = state_selected

    return arr_states


@callback(
    Output({"type": "cell", "index": ALL}, "children"),
    Input("arr_states", "data"),
)
def style_update(arr_states):
    # flatten, matching the ordering of the cells
    # this is a mess but whatever
    style_names = np.array(arr_states).transpose()[::-1, :].flatten()

    # look up styles
    texts = [IMAGES[style_name] for style_name in style_names]

    return texts


@callback(
    Output("page", "children"), Input("n_cols", "value"), Input("n_rows", "value")
)
def change_dims(n_cols, n_rows):
    """Save to a JSON"""
    return make_layout(n_cols, n_rows)


@callback(Input("save", "n_clicks"), State("arr_states", "data"))
def save(_, arr_states):
    """Save to a JSON"""
    d = convert_to_dict(arr_states)
    filename = "boards/board.json"
    save_dict_to_json(d, filename)


if __name__ == "__main__":
    app.run(debug=True)
