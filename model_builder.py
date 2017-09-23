from pandas import DataFrame, read_csv
import pandas as pd 

file = r"data/MsiaAccidentCases.xlsx"
df = pd.read_excel(file)

print("rows, columns: " + str(df.shape))
print(df[df["Title Case"].isnull()])

#drop all rows with no data
df=df.dropna(how='all')
print("rows, columns: " + str(df.shape))
df[df["Title Case"].isnull()]


#drop rows having Cause = TEST DATA
df=df[df["Cause"]!="TEST DATA"]
print("rows, columns: " + str(df.shape))
df[df["Title Case"].isnull()]