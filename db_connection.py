import pyodbc

def get_connection():
    """
    Establishes and returns a SQL Server database connection.
    """
    conn = pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=TATHAGATA\\SQLEXPRESS;"
        "DATABASE=MyDatabase;"
        "Trusted_Connection=yes;"
    )
    return conn