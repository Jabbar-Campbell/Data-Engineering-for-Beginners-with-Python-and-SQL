#
# Class Project - Contact Database - SQLite3 Database Processing using Functions
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
      dbcon = sqlite3.connect('test.db')

   except:
      print("Open database failed")
      
   return

#
# Create a Table
#

def CreateTable():
   
   import sqlite3

   try:
      dbcon = sqlite3.connect('test.db')

   except:
      print("Open database failed")

   try:
      dbcon.execute('''CREATE TABLE CONTACT
            (NAME           TEXT PRIMARY KEY  NOT NULL,
            AGE             INT    NOT NULL,
            STREET_ADDRESS         CHAR(40),
            CITY                   CHAR(15),
            STATE                  CHAR(2),
            NOTES                  CHAR(150));''')
   except:
      print("Table exists")

   dbcon.close()

   return

#
# When the above program is executed, it will create the COMPANY table in your test.db and it will display the following messages -
# Opened database successfully
# Table created successfully
#

#
# Insert Records
#

def InsertRecords():

   import sqlite3

   try:
      dbcon = sqlite3.connect('test.db')
   except:
      print("Open database failed")

   try:
      name = 'Thomas'
      age  = 62
      street_address = '7584 James Blvd'
      city = 'Anchorage'
      state = 'AK'
      notes = "These are my notes"

      dbcon.execute("INSERT INTO CONTACT (NAME, AGE, STREET_ADDRESS, CITY, STATE, NOTES) VALUES (?, ?, ?, ?, ?, ?)", (name,age,street_address,city,state,notes))
      dbcon.commit()
      print("Insert Using Variables SUCCEEDED")
   except:
      print("Insert Using Variables assigned to Thomas FAILED")

   try:
      dbcon.execute("INSERT INTO CONTACT (NAME,AGE,STREET_ADDRESS,CITY, STATE) VALUES ('Peter', 62, '7534 Shastain', 'Yorbalinda', 'CA' )")

      dbcon.execute("INSERT INTO CONTACT (NAME,AGE,STREET_ADDRESS,CITY, STATE) \
         VALUES ('Paul Sluggle', 33, '6456 Fortuna Ave', 'Allentown', 'PA')")

      dbcon.execute("INSERT INTO CONTACT (NAME,AGE,STREET_ADDRESS,CITY, STATE) \
         VALUES ('Mary Coporkney', 99, '8432 Shastel Ln', 'Austin', 'TX' )")

      dbcon.execute("INSERT INTO CONTACT (NAME,AGE,STREET_ADDRESS,CITY, STATE) \
         VALUES ('June Fritzenheim', 21, '1610 Ace Ventura Blvd', 'Topeka', 'KS')")

      dbcon.execute("INSERT INTO CONTACT (NAME,AGE,STREET_ADDRESS,CITY, STATE) \
         VALUES ('Roger Snooks', 21, '1400 Carls Ln', 'Topeka', 'KS')")

#      dbcon.execute("INSERT INTO CONTACT (NAME,AGE,STREET_ADDRESS,CITY, STATE) \
#         VALUES ('Roger Snooks', 42, '3201 Friars Ave', 'Topeka', 'KS')")

      dbcon.commit()

   except:
      print("Insert failed")

   return

def InsertNewRecord():

   import sqlite3

#   prev_name = name_entered

   name_entered = e2_value.get()

   if name_entered == "":
      return
   
   address_entered = e3_value.get()
   city_entered = e4_value.get()
   state_entered = e5_value.get()
   notes_entered = e6_value.get()

#   name_entered = "Antonio"
#   address_entered = "472389 Jims Blvd"
#   city_entered = "San Diego"
#   state_entered = "CA"
#   notes_entered = "These are my notes"

   age = 0 # Age not included elsewhere - Add if you wish!
   
   print("InsertNewRecord Function Entered")
   print("name inserted: ",name_entered)
   print("address inserted: ",address_entered)
   print("city inserted: ",city_entered)
   print("state inserted: ",state_entered)
   print("notes inserted: ",notes_entered)
   print("InsertNewRecord Function Exited")

   try:
      dbcon = sqlite3.connect('test.db')
   except:
      print("Open database failed")

   try:
      dbcon.execute("INSERT INTO CONTACT (NAME, AGE, STREET_ADDRESS, CITY, STATE, NOTES) VALUES (?, ?, ?, ?, ?, ?)", (name_entered,age,address_entered,city_entered,state_entered,notes_entered))
      dbcon.commit()
      print("Insert Using Variables SUCCEEDED")

   except:
      print("Insert Using Variables FAILED possibility due to the record already existing")

      dbcon.execute("DELETE from CONTACT where NAME = ?",(name_entered,)) 
      dbcon.execute("INSERT INTO CONTACT (NAME, AGE, STREET_ADDRESS, CITY, STATE, NOTES) VALUES (?, ?, ?, ?, ?, ?)", (name_entered,age,address_entered,city_entered,state_entered,notes_entered))
      dbcon.commit()
      print("Insert Using Variables SUCCEEDED")

#     return

#
# Select Records
#

def SelectRecord(name):

   import sqlite3

   dbcon = sqlite3.connect('test.db')
   print ("Opened database successfully")

#   name = 'Paul Sluggle'
   query = "SELECT name, street_address, city, state, notes from CONTACT WHERE name = '" + name + "'"
   print(query)


   try:
      cursor = dbcon.execute(query)
      for row in cursor:
         print("Name:    ", row[0])
         print("Address: ", row[1])
         print("City:    ", row[2])
         print("State:   ", row[3])
         print("Notes:   ", row[4], "\n")
         print("Noogies")

      background = "blue"
      foreground = "white"

   except:
      print("Query Failed")

   print("Select Record - Operation done successfully")

#   dbcon.close()

   Populate_Data("", "", "", "", "") # Clear the Fields on the Form If Theres an Error

   return(row[0], row[1], row[2], row[3], row[4])

def SelectAllRecords(aList):

   import sqlite3

   cntacts = []
   
   dbcon = sqlite3.connect('test.db')
   print ("Opened database successfully")

#   cursor = dbcon.execute("SELECT name, street_address, city, state from CONTACT WHERE name = '*'")
   cursor = dbcon.execute("SELECT name, street_address, city, state from CONTACT")
   for row in cursor:
      print("Name:    ", row[0])
      print("Address: ", row[1])
      print("City:    ", row[2])
      print("State:   ", row[3], "\n")
      print("Noogies")
      aList.append(row[0])
      print("aList = ",aList)
      
#      cntacts.append('Peter')
#      cntacts.append(row[0])

   dbcon.commit()
   print("SelectAllRecords Complete")

   dbcon.close()
 
   return aList
#
# Update Records 
#

def UpdateRecords():

   import sqlite3

   dbcon = sqlite3.connect('test.db')
   print("Opened database successfully")

   dbcon.execute("UPDATE CONTACT set NOTES = 'Our Primary Contact' WHERE NAME = 'Peter'")
   dbcon.commit
   print("Total number of rows updated :", dbcon.total_changes)

   cursor = dbcon.execute("SELECT name, street_address, city, state from CONTACT")
   for row in cursor:
      print("Name:    ", row[0])
      print("Address: ", row[1])
      print("City:    ", row[2])
      print("State:   ", row[3], "\n")

   print("Operation done successfully")
   dbcon.close()

   return

def DeleteRecord():

   import sqlite3

   try:
       dbcon = sqlite3.connect('test.db')
       print("Open Database SUCCESSFUL")
   except:
       print("Open Database FAILED")

   name_entered = e2_value.get()

   print("Exit of DELETE")


#   dbcon.execute("DELETE from CONTACT where NAME = 'Cribnits Alamo' ;")

   try:

      print("Name_Entered = ",name_entered)
      print("Length of name_entered is: %d",len(name_entered))

#
# **** THIS PARAGRAPH WORKS BUT SHOULD NOT BE USED DUE TO SECURITY CONCERNS OVER SQL INJECTION ATTACKS ****
#
#   sql="DELETE from CONTACT where NAME = '"+name_entered+"'"  
#   print(sql)
#   dbcon.execute(sql)


#
# NOTE the comma ',' after name_entered - (name_entered,) - It's necessary to send a tuple
# NOTE the question mark '?' is used as a place marker to substitute the value of name_entered specifed as a tuple using (name_entered,)
# NOTE this method is safer and guards against SQL injection attacks
#
      dbcon.execute("DELETE from CONTACT where NAME = ?",(name_entered,)) 


      print("Delete Using Variables SUCCEEDED")

   except:
      print("Delete Using Variables FAILED")

   dbcon.commit() # Commit the changes - in this case a deletion - to the database

   print("Total number of rows deleted :", dbcon.total_changes)

   dbcon.close()

#   return

#
# CALL Functions in Order to Perform Processing
#

DBConnect()
CreateTable()
InsertRecords()
# SelectRecord()
UpdateRecords()
# DeleteRecords()
#SelectAllRecords()

#
from tkinter import *
from tkinter import Text

# global e3_value


win=Tk()

win.title("Business Contact Database")

win.geometry("500x280")

def Manually_Typed_Search():
    name_entered = e2_value.get()

    name, address, city, state, notes = SelectRecord(name_entered)

    print("The Named is: {}".format(name))

    print("The Address is: {}".format(address))

    print("The City is: {}".format(city))

    print("The State is: {}".format(state))

    print("The State is: {}".format(notes))

    Populate_Data(name, address, city, state, notes)


def List_Search():
    def List_Select(evt):

#      list_box_name = cntlist.curselection()
       list_box_name = str(cntlist.get(ANCHOR))

       print("index==",list_box_name)       

       print("In List_Select Function")

#      e2_value=StringVar()
       e2_value.set(list_box_name)
       e2=Entry(win,text=e2_value)
       e2.grid(row=0,column=1)
       print("***********************",e2_value)


       Manually_Typed_Search()
       win2.destroy()

    win2 = Tk()
    Sb1 = Scrollbar(win2)
    Sb1.pack(fill = Y, side = RIGHT)

    
    global cntlist
    cntlist = Listbox(win2, width=20, height=10, yscrollcommand = Sb1.set, selectmode=SINGLE) # Select only one single value
    cntlist.bind('<<ListboxSelect>>',List_Select)


#    return


        
# Generate Contact ListBox Data for Demonstration Purposes
#    cntacts = ['Peter','Bob Fenway','Fred Porkenstein','Roy Smickler', 'Kryton Semiphor', 'Bill Binney', 'Bunny Balihoo']

#    DATABASE SEARCH to POPULATE ALL NAMES

    cntacts=[]

    SelectAllRecords(cntacts)

    print("List_Search - cntacts = ",cntacts)

    cntacts.sort() # Sort List of Contacts to make location of specific contacts easier when scrolling through the list

# Insert the Contact Data into the cntlist ListBox
    for cntact in cntacts:
        cntlist.insert(END, cntact)
   
# Insert Additional Generated Contact Data into the cntlist ListBox so we have a little more data to play with.
#
#    for lineNum in range(10):
#        cntlist.insert(END, "More Names Here " + str(lineNum))

#    select = cntlist.get(cntlist.curselection())

    cntlist.pack(fill = BOTH, side = LEFT)
    Sb1.config(command = cntlist.yview)
            
    win2.mainloop()

def Populate_Data(name, address, city, state, notes):
#    e3_value=StringVar()
    e3_value.set(address)
    e3=Entry(win,text=e3_value)
    e3.grid(row=1,column=1)
    
#    e4_value=StringVar()
    e4_value.set(city)
    e4=Entry(win,text=e4_value)
    e4.grid(row=2,column=1)

#    e5_value=StringVar()
    e5_value.set(state)
    e5=Entry(win,text=e5_value)
    e5.grid(row=3,column=1)

#    e6_value=StringVar()
    e6_value.set(notes)
    e6=Entry(win,text=e6_value)
    e6.grid(row=4,column=1,padx=35,pady=25)

    Display_Labels()

def Display_Labels():
    e3=Label(win,text="Address")
    e3.grid(row=1,column=0)
      
    e4=Label(win,text="City")
    e4.grid(row=2,column=0)

    e5=Label(win,text="State")
    e5.grid(row=3,column=0)

    e6=Label(win,text="Notes:")
    e6.grid(row=4,column=0)

#
# Initialize 'e' variables associated with tkinter form fields
#

e3_value=StringVar()
e4_value=StringVar()
e5_value=StringVar()
e6_value=StringVar()

#
# Initialize tkinter form by setting the fields to blanks
#

Populate_Data(" ", " ", " ", " ", " ") # Clear all fields on the form

#
# Call the Display_Labels function to setup the fielf labels on the form
#

Display_Labels()

e1=Label(win,text="Enter Name")
e1.grid(row=0,column=0,ipadx=35,ipady=25)

e2_value=StringVar()
e2=Entry(win,textvariable=e2_value) # Prepare Enter Name field on form with value (name) to be Searched for
e2.grid(row=0,column=1)

     
b1=Button(win,text="Search ",command=Manually_Typed_Search)
b1.grid(row=0,column=2,padx=35,pady=25)

b2=Button(win,text="  List   ",command=List_Search)
b2.grid(row=1,column=2,ipadx=5,ipady=2)

b3=Button(win,text="Save",command=InsertNewRecord)
b3.grid(row=3,column=2,ipadx=5,ipady=2)

b4=Button(win,text="Delete",command=DeleteRecord)
b4.grid(row=4,column=2,ipadx=5,ipady=2)

#
# Add a new function to provide a great new feature to this program and uncomment the lines below to make use of the button defined here.
# Change YourFunctionHere to the name of your new function so the button will activate your feature when pressed.
#
# b5=Button(win,text="Delete",command=YourFunctionHere)
# b5.grid(row=5,column=2,ipadx=5,ipady=2)

win.mainloop()



