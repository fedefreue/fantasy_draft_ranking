from flask import Flask, render_template, request, session
from flask_session import Session
import yahoo_fantasy_api as yfa
import pandas as pd
import numpy as np

import connect
import playerData

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

points = {
    "Pass Yds": 0.04,
    "Pass TD": 4,
    "Int": -1,
    "Rush Att": 0.1,
    "Rush TD": 6,
    "Ret TD": 6,
    "Rec": 1,
    "Rec Yds": 0.1,
    "Rec TD": 6,
    "2-PT": 2,
    "Fum Lost": -2,
    "Fum Ret TD": 2,
}
seasonList = [2021, 2022, 2023]
positionList = ["RB", "QB", "WR", "TE"]

yfa_connection = None


def model_train(masterTable: pd.DataFrame):
    features = masterTable.drop(columns=["position_type", "player_id", "Points"])
    y_train = masterTable["Points"]

    features = features.to_numpy()
    y_train = y_train.to_numpy()

    model = sklearn.linear_model.LinearRegression().fit(features, y_train)
    statusText = "R2: " + str(model.score(features, y_train))
    # plt.scatter(self.model.predict(features), y_train)
    # plt.show()


@app.route("/", methods=["GET", "POST"])
def route_home():
    if request.form.get("connect") == "connect":
        session["yfa_connection"] = connect.connect("private.json")
        return render_template(
            "home.html", buttonvalue=session.get("yfa_connection", None)
        )
    else:
        return render_template("home.html", buttonvalue="Error")


@app.route("/data/", methods=["GET", "POST"])
def route_data():
    if request.form.get("gen_table") == "ok":
        league_code = "nfl.l.209930"  # Get from a textbox
        session["league_id"] = yfa.League(session.get("yfa_connection"), league_code)
        data_table = playerData.generateTrainingTable(
            seasonList, session.get("league_id", None), points, positionList
        )
        return render_template(
            "data.html", tables=[data_table.to_html(classes="data", header="true")]
        )
    else:
        return render_template("data.html")


@app.route("/model/", methods=["GET", "POST"])
def route_model():
    if request.form.get("train") == "train":
        return render_template("model.html")
    else:
        return render_template("model.html")


@app.route("/predict/")
def route_predict():
    return render_template("predict.html")


if __name__ == "__main__":
    app.run()
