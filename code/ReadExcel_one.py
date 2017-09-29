# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import xlrd
import re
import string
#Natural Language tool kit (nltk)
import nltk
from nltk import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

workbook = xlrd.open_workbook("C:\\Classes\\Sem2_jul_to_dec_2017\\TextMining\\Project\\MsiaAccidentCases.xlsx")
worksheet = workbook.sheet_by_index(0)

#sheet names, count
print("the totla no of sheets===>: {0}".format(workbook.nsheets))
print("the totla sheets names===>: {0}".format(workbook.sheet_names()))
print("the totla no of rows===>: {0}".format(worksheet.nrows))
print("the totla no of columns===>: {0}".format(worksheet.ncols))

total_rows = worksheet.nrows
total_cols = worksheet.ncols
table = list()
record = list()

#Read only specific category values
for x in range(total_rows):
    if worksheet.cell(x,0).value == 'Other':
        record.append(worksheet.cell(x,0).value)
        record.append(worksheet.cell(x,1).value)
        record.append(worksheet.cell(x,2).value)
        table.append(record)
        record = []
        x += 1
    
    
#Read all excel values
#for x in range(total_rows):
#    for y in range(total_cols):
#        record.append(worksheet.cell(x,y).value)
#    table.append(record)
#    record = []
#    x += 1
          
print (table)    


masia_str = ' '.join(str(v).strip('[]().,\'') for v in table)
print(masia_str)
words = word_tokenize(masia_str) 
print(len(words))
print(words)

stop_words = set(stopwords.words("english")) 

stop_words.update(',')
stop_words.update("'")
stop_words.update(')')
stop_words.update('(')
stop_words.update('.')  
print(len(stop_words))

removed_stop_words = []
for w in words:
    if w.lower() not in stop_words:
        removed_stop_words.append(w.lower())
print(removed_stop_words)
print(len(removed_stop_words))

#print(removed_stop_words)

# Back to text preprocessing: remove punctuations
tokens_nop = [ t for t in removed_stop_words if t not in string.punctuation ]
len(tokens_nop)
len(set(tokens_nop))


#Utility Functions
def hasNumbers(inputString):
     return bool(re.search(r'\d', inputString))
 
def hasSpecialChar(inputString):
    return bool(re.search(r'[\']', inputString))

final_words = []
for fw in removed_stop_words:
    if hasNumbers(fw):
         print("Number===>"+fw)
    elif hasSpecialChar(fw):
        print("special===>"+fw)
    else:
        final_words.append(fw)
        
unique = set(final_words)
len(unique)      
print(final_words)
print(len(final_words))  


# The most popular stemmer
porter = nltk.PorterStemmer()
stemmer_fall_words=[ porter.stem(t) for t in final_words ] 
print(stemmer_fall_words)
print(len(stemmer_fall_words))

# Frequency distribution of the words
#final_words.count('died')
fd = nltk.FreqDist(stemmer_fall_words , )
fd.most_common(10)
fd.plot(10)


