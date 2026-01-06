import pandas as pd
from db_connection import get_connection

def load_base_data():
    """
    Loads the OLA Cleaned table from SQL Server into a Pandas DataFrame.
    """
    conn = get_connection()
    query = "SELECT * FROM [dbo].[OLA Cleaned];"
    df = pd.read_sql(query, conn)
    conn.close()
    return df