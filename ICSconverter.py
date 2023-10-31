# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 11:02:19 2023

@author: iru-ra2
"""

import pandas as pd
import re

#Load Excel worksheet
filepath ="C:/Users/Desktop/BCGRISE.xlsx"
savetarget_path = "C:/Users/Desktop/calendar.csv"
df = pd.read_excel(filepath,engine = 'openpyxl')

print(df.head(10))

def excel_to_ics(a):

    # Start the ICS content
    ics_content = "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//BCGRISE Calendar/EN\nCALSCALE:GREGORIAN \n"

    # Loop through rows in the DataFrame and generate ICS events
    for _, row in a.iterrows():
        Start_Date, Start_Time, End_Date, End_Time, Summary, Description = row
        f_Start_Date = Start_Date.strftime('%Y%m%d').strip()
        f_Start_Date =re.sub(r'\-','', str(f_Start_Date))
        Start_Time =re.sub(r'\:','', str(Start_Time)).strip()
        End_Time   =re.sub(r'\:','', str(End_Time)).strip()
        ics_content += "BEGIN:VEVENT\n"
        ics_content += f"DTSTART:{f_Start_Date}T{Start_Time}\n"
        ics_content += f"DTEND:{f_Start_Date}T{End_Time}\n"
        ics_content += f"SUMMARY:{Summary}\n"
        ics_content += f"DESCRIPTION:{Description}\n"
        ics_content += "END:VEVENT\n"
        ics_content += "\n"

    ics_content += "END:VCALENDAR"

   # Save to ICS file
    with open("output.ics", "w") as f:
        f.write(ics_content)

    return ics_content

# Call the function
ics_output = excel_to_ics(df)
print(ics_output)
with open(savetarget_path, 'w') as file:
    file.write(ics_output)

