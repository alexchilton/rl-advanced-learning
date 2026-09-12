"""Fill the Advanced SQL exercise notebooks.

Answers come from learntools/sql_advanced/ex{1..4}.py, not from reading the
question. Two of these are graded by running the query against BigQuery and
comparing row counts to a reference result, so a query that merely looks right
is not good enough.

    python kaggle/advanced-sql/build_notebooks.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

EX1_Q2 = '''# Your code here
q_and_a_query = """
                SELECT q.owner_user_id AS owner_user_id,
                    MIN(q.creation_date) AS q_creation_date,
                    MIN(a.creation_date) AS a_creation_date
                FROM `bigquery-public-data.stackoverflow.posts_questions` AS q
                    FULL JOIN `bigquery-public-data.stackoverflow.posts_answers` AS a
                ON q.owner_user_id = a.owner_user_id 
                WHERE q.creation_date >= '2019-01-01' AND q.creation_date < '2019-02-01' 
                    AND a.creation_date >= '2019-01-01' AND a.creation_date < '2019-02-01'
                GROUP BY owner_user_id
                """

# Check your answer
q_2.check()
'''

EX2_Q1 = '''# Fill in the blank below
avg_num_trips_query = """
                      WITH trips_by_day AS
                      (
                      SELECT DATE(trip_start_timestamp) AS trip_date,
                          COUNT(*) as num_trips
                      FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                      WHERE trip_start_timestamp > '2016-01-01' AND trip_start_timestamp < '2016-04-01'
                      GROUP BY trip_date
                      ORDER BY trip_date
                      )
                      SELECT trip_date,
                          AVG(num_trips)
                          OVER (
                               ORDER BY trip_date
                               ROWS BETWEEN 3 PRECEDING AND 3 FOLLOWING
                               ) AS avg_num_trips
                      FROM trips_by_day
                      """

# Check your answer
q_1.check()
'''

EX2_Q3 = '''# Fill in the blanks below
break_time_query = """
                   SELECT taxi_id,
                       trip_start_timestamp,
                       trip_end_timestamp,
                       TIMESTAMP_DIFF(
                           trip_start_timestamp, 
                           LAG(trip_end_timestamp, 1) 
                               OVER (
                                    PARTITION BY taxi_id 
                                    ORDER BY trip_start_timestamp), 
                           MINUTE) as prev_break
                   FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
                   WHERE DATE(trip_start_timestamp) = '2013-10-03' 
                   """

# Check your answer
q_3.check()
'''

EX3_Q2 = '''# Fill in the blank
# UNNEST flattens the repeated `language` column, so a row is repeated once per
# language the repo uses. Six rows come out of the sample.
num_rows = 6

# Check your answer
q_2.check()
'''

EX4_Q1 = '''# Fill in your answer
# Query 3 is the one worth optimising: it scans the whole table rather than
# restricting to the pet in question.
query_to_optimize = 3

# Check your answer
q_1.check()
'''

NOTEBOOKS = {
    "ex1": ("exercise-joins-and-unions.ipynb", {13: ("q_and_a_query", EX1_Q2)}),
    "ex2": ("exercise-analytic-functions.ipynb", {
        6: ("avg_num_trips_query", EX2_Q1),
        12: ("break_time_query", EX2_Q3),
    }),
    "ex3": ("exercise-nested-and-repeated-data.ipynb", {15: ("num_rows = ____", EX3_Q2)}),
    "ex4": ("exercise-writing-efficient-queries.ipynb", {4: ("query_to_optimize = ____", EX4_Q1)}),
}


def build(name, filename, edits):
    nb = json.loads((HERE / "sources" / filename).read_text())
    for index, (needle, replacement) in sorted(edits.items()):
        cell = nb["cells"][index]
        text = "".join(cell["source"])
        if cell["cell_type"] != "code" or needle not in text or "____" not in text:
            raise SystemExit(f"{name}: cell {index} is not the stub for {needle!r}")
        cell["source"] = replacement.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
    out = HERE / name / filename
    out.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {out.relative_to(HERE.parents[1])}  ({len(edits)} cells)")


def main():
    for name, (filename, edits) in NOTEBOOKS.items():
        build(name, filename, edits)


if __name__ == "__main__":
    main()
