import mysql.connector

# tạo đối tượng connection
myconn = mysql.connector.connect(host = "localhost", user = "root",
    passwd = "root" , database = "chatdb")

# tạo đối tượng cursor
cur = myconn.cursor()

try:
    dbs = cur.execute("show databases")
except:
    myconn.rollback()
for x in cur:
    print(x)

myconn.close()