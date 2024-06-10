#
# SQLite3 Database Processing using Functions
#
#
# Copyright 2019 - Primordius LLC - All Rights Reserved
#
# PROGRAM: SQLite Functions PythonEx3_V36
# connect / Create DB if necessary
# Insert / Update / Delete / Search
# Commit Changes
# Close DB
#
# Functions Created and Executed:
#     DBConnect()
#     CreateTable()
#     InsertRecords()
#     SelectRecords()
#     UpdateRecords()
#     DeleteRecords()

#
# Connect To Database - If the database does not exist, it will be created.
#

def DBConnect():
   import sqlite3

   try:
      dbcon = sqlite3.connect('test_initial.db')

   except:
      print("Open database failed")
      
   return

#
# Create a Table
#

def CreateTable():
   
   import sqlite3

   try:
      dbcon = sqlite3.connect('test_initial.db')

   except:
      print("Open database failed")

   try:
      dbcon.execute('''CREATE TABLE CONTACT
            (ID INT PRIMARY KEY    NOT NULL,
            NAME            TEXT   NOT NULL,
            AGE             INT    NOT NULL,
            STREET_ADDRESS         CHAR(40),
            CITY                   CHAR(15),
            STATE                  CHAR(2),
            NOTES                  CHAR(150))''')
   except:
      print("Table exists")

   dbcon.close()

   return

#
# When the above program is executed, it will create the COMPANY table in your test_initial.db and it will display the following messages -
# Opened database successfully
# Table created successfully
#

#
# Insert Records
#

def InsertRecords():

   import sqlite3

   try:
      dbcon = sqlite3.connect('test_initial.db')
   except:
      print("Open database failed")

   try:
      dbcon.execute("INSERT INTO CONTACT (ID,NAME,AGE,STREET_ADDRESS,CITY, STATE, NOTES) VALUES (1, 'Peter', 62, '7534 Shastain', 'Yorbalinda', 'CA', 'Notes Go Here' )")

      dbcon.execute("INSERT INTO CONTACT (ID,NAME,AGE,STREET_ADDRESS,CITY, STATE) \
         VALUES (2, 'Paul Sluggle', 33, '6456 Fortuna Ave', 'Allentown', 'PA')")

      dbcon.execute("INSERT INTO CONTACT (ID,NAME,AGE,STREET_ADDRESS,CITY, STATE) \
         VALUES (3, 'Mary Coporkney', 99, '8432 Shastel Ln', 'Austin', 'TX' )")

      dbcon.execute("INSERT INTO CONTACT (ID,NAME,AGE,STREET_ADDRESS,CITY, STATE) \
         VALUES (4, 'June Fritzenheim', 21, '1610 Ace Ventura Blvd', 'Topeka', 'KS')")

      dbcon.commit()

   except:
      print("Insert failed")

   return
#
# Select Records
#

def SelectRecord():

   import sqlite3

   dbcon = sqlite3.connect('test_initial.db')
   print ("SelectRecord()")

   cursor = dbcon.execute("SELECT id, name, street_address, city, state, notes from CONTACT WHERE name = 'Paul Sluggle'")
   for row in cursor:
      print("ID:      ", row[0])
      print("Name:    ", row[1])
      print("Address: ", row[2])
      print("City:    ", row[3])
      print("State:   ", row[4])
      print("Notes:   ", row[5], "\n")
      print("End of Record")

   dbcon.close()

   return

#
# Update Records 
#

def UpdateRecords():

   import sqlite3

   dbcon = sqlite3.connect('test_initial.db')
   print("UpdateRecords()")

   dbcon.execute("UPDATE CONTACT set NOTES = 'Our Primary Contact' where ID = 1")
   dbcon.commit
   print("Total number of rows updated :", dbcon.total_changes)

   cursor = dbcon.execute("SELECT id, name, street_address, city, state, notes from CONTACT")
   for row in cursor:
      print("ID:      ", row[0])
      print("Name:    ", row[1])
      print("Address: ", row[2])
      print("City:    ", row[3])
      print("State:   ", row[4])
      print("Notes:   ", row[5], "\n")

   print("Operation Executed")
   
   dbcon.close()

   return
#
# Delete Record
#

def DeleteRecords():

   import sqlite3

   dbcon = sqlite3.connect('test_initial.db')
   print("DeleteRecords()")

   dbcon.execute("DELETE from CONTACT where ID = 1;")
   dbcon.execute("DELETE from CONTACT where ID = 2;")
   dbcon.execute("DELETE from CONTACT where ID = 3;")
   dbcon.execute("DELETE from CONTACT where ID = 4;")

   dbcon.commit()
   
   print("Total number of rows deleted :", dbcon.total_changes)

   cursor = dbcon.execute("SELECT id, name, street_address, city, state, notes from CONTACT")
   for row in cursor:
      print("ID:      ", row[0])
      print("Name:    ", row[1])
      print("Address: ", row[2])
      print("City:    ", row[3])
      print("State:   ", row[4])
      print("State:   ", row[5], "\n")

   print("Operation Executed")
   
   dbcon.close()

   return

#
# CALL Functions in Order to Perform Processing
#

DBConnect()
CreateTable()
InsertRecords()
SelectRecord()
UpdateRecords()
DeleteRecords()

