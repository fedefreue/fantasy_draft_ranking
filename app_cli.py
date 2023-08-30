import data_prep
import model
import sqlite3

layer1 = 13
layer2 = 6
current_year = 2022
years_to_train = 10
positions = ["RB", "QB", "WR", "TE"]

dbConnection = sqlite3.connect("db.sqlite3")

with open("schema.sql", "r") as sql_file:
    sql_script = sql_file.read()
    cur = dbConnection.cursor()
    cur.executescript(sql_script)
    # cur.commit()
    cur.close()

# Check if the initializated bit exists
if (
    dbConnection.execute("SELECT COUNT(*) FROM dbInitialize WHERE bool = 1").fetchall()
) is not None:
    print("Database has been initialized")
else:
    raise Exception("Database has not been initialized")

# For each value in dataYears, scrape the data into a raw_data table
# dbConnection.execute("DROP TABLE IF EXISTS rawData")
# dbConnection.commit()

data_years = data_prep.gen_year_list(current_year, years_to_train)
data_prep.raw_dl(data_years, dbConnection)
data_table = data_prep.format(dbConnection)

model.optimize_set(
    db_connection=dbConnection,
    layer1=layer1,
    layer2=layer2,
    staging_table_name="features",
    save_model=1,
    file_name="debug",
    by_position=1,
    positions=["RB", "QB", "WR", "TE"],
    verbose=1,
)
