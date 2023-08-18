from flask import Flask, render_template, request, session
from flask_session import Session
import yahoo_fantasy_api as yfa
import pandas as pd
import numpy as np
import sklearn
import data_prep
import sqlite3

import connect
import model

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
thisYear = 2021
yearsToTrain = 3
positionList = ["RB", "QB", "WR", "TE"]

yfa_connection = None
db_connection = None


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


"""
league_code = "nfl.l.209930"  # Get from a textbox
session["league_id"] = yfa.League(session.get("yfa_connection"), league_code)
"""


@app.route("/data/", methods=["GET", "POST"])
def route_data():
    if request.form.get("db_init") == "Initialize DB":
        db_connection = sqlite3.connect("db.sqlite3")
        with open("schema.sql", "r") as sql_file:
            sql_script = sql_file.read()
        cur = db_connection.cursor()
        cur.executescript(sql_script)
        # cur.commit()
        cur.close()
        return render_template(
            "data.html", db_status_flag="Initialized to DB at db.sqlite3"
        )
    elif request.form.get("db_connect") == "Connect to DB":
        db_connection = sqlite3.connect("db.sqlite3")
        if (
            db_connection.execute(
                "SELECT COUNT(*) FROM dbInitialize WHERE bool = 1"
            ).fetchall()
        ) is not None:
            return render_template(
                "data.html", db_status_flag="Connected to DB at db.sqlite3"
            )
        else:
            raise Exception("Not connected to DB or it has not been initialized")
    elif request.form.get("gen_table") == "Scrape Data":
        db_connection = sqlite3.connect("db.sqlite3")
        data_years = data_prep.data_gen_year_list(thisYear, yearsToTrain)
        data_prep.data_rawdl(data_years, db_connection)
        data_table = data_prep.data_format(db_connection)
        debug_data_table = pd.read_sql_query(
            "SELECT * FROM features ORDER BY points DESC LIMIT 100;", db_connection
        )
        return render_template(
            "data.html",
            db_status_flag="Data scrape completed.",
            tables=[
                debug_data_table.to_html(classes="table table-striped", header="true")
            ],
        )
    else:
        return render_template("data.html")


@app.route("/model/", methods=["GET", "POST"])
def route_model():
    if request.form.get("model_train") == "Train Model":
        db_connection = sqlite3.connect("db.sqlite3")
        model_arch = model.model_ranks(db_connection, "features", 1)
        return render_template("model.html", arch_debug = str(model_arch))
    else:
        return render_template("model.html")


@app.route("/predict/")
def route_predict():
    return render_template("predict.html")


if __name__ == "__main__":
    app.run()
