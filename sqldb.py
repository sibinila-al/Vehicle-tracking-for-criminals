import mysql.connector

# Connect to the MySQL database
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="projectx"
)

# Create a cursor object to interact with the database
cursor = conn.cursor()

# Create users table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users(
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(50) UNIQUE,
        password VARCHAR(255)
    )
''')

# Create detections table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS detections(
        id INT AUTO_INCREMENT PRIMARY KEY,
        license_plate VARCHAR(255),
        date DATE,
        time TIME,
        location VARCHAR(255),
        monitoring_start DATETIME,
        monitoring_end DATETIME,
        image LONGBLOB
    )
''')

# Create camera_history table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS camera_history(
        id INT AUTO_INCREMENT PRIMARY KEY,
        action VARCHAR(255),
        timestamp DATETIME,
        duration INT
    )
''')

conn.commit()
conn.close()