import sqlite3 as sql
import pandas as pd
import sys
import os
from src.utils import create_db_engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import create_db_engine
data_path = "./Data/hotel_bookings.csv"

def read_csv(file_path: str):
    try:
        # Read csv file into dataframe
        df = pd.read_csv(file_path)
        print(f"Successfully read csv file at {file_path}")
        return df
    except pd.errors.EmptyDataError:
        print(f"Error reading csv file: {file_path} is empty")
        return None
    except pd.errors.ParserError:
        print(f"Error reading csv file: {file_path} is not a valid csv file")
        return None


def write_to_db(df: pd.DataFrame, table_name: str, conn: sql.Connection):
    try:
        # Write dataframe to database
        df.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Successfully wrote dataframe to database")
    except sql.Error as e:
        print(f"Error writing dataframe to database: {e}")



df = read_csv(data_path)
if df is not None:
    conn = create_db_engine()
    write_to_db(df, "hotel_bookings", conn)
