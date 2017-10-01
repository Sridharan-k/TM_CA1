from pandas import DataFrame, read_csv
import pandas as pd 
from nltk import FreqDist, word_tokenize, SnowballStemmer, WordNetLemmatizer
import string
from nltk.corpus import stopwords
from sklearn import metrics, tree
#, SGDClassifier
from sklearn.naive_bayes import MultinomialNB

MY_CASES_FILE=r"C:\githome\ivarunkumar\KE5205-TextMining-CA\data\MsiaAccidentCases.xlsx"
custom_stopwords = ["victim", "year", "morning", "afternoon", ]
stop = stopwords.words('english') + custom_stopwords
snowball = SnowballStemmer('english')
wnl = WordNetLemmatizer()



def preprocessDocument(text) :
    toks = word_tokenize(text)
    toks = [ t.lower() for t in toks if t not in string.punctuation ]
    toks = [t for t in toks if t not in stop ]
    toks = [ wnl.lemmatize(t) for t in toks ]
    out= " ".join(toks)
    return out

'''
Read the input file and clean the data found for preprocessing.
'''
def loadFileAndProcess(path) :
    dataFrame = pd.read_excel(path, names = ["Cause", "Title Case", "Summary Case"])
    #drop all rows with no data
    dataFrame=dataFrame.dropna(how='all')
    print("rows, columns: " + str(dataFrame.shape))
    dataFrame[dataFrame["Title Case"].isnull()]
    dataFrame.groupby('Cause').describe()
    #Count the length of each document
    length=dataFrame['Summary Case'].apply(len)
    dataFrame=dataFrame.assign(Length=length)
    
    #Plot the distribution of the document length for each category
    import matplotlib.pyplot as plt
    dataFrame.hist(column='Length',by='Cause',bins=10)
    plt.show()
    
    #Apply the function on each document
    dataFrame['Text'] = dataFrame['Summary Case'].apply(preprocessDocument)
    #dataFrame.head()
    return dataFrame

#Build a pipeline: Combine multiple steps into one
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
#def runPipeline() :
#    text_clf = Pipeline([('vect', CountVectorizer()),  
#                         ('tfidf', TfidfTransformer()),
#                         ('clf', MultinomialNB()),
#                         ])
#    return text_clf


#def buildModel() :
#    
#def __main__() :
#    generateDTM(MY_CASES_FILE)
#    

#def generateDTM(path) : 
#    df = loadFileAndClean(path)
#    for d in df :
#        print (d[2])

df = loadFileAndProcess(MY_CASES_FILE)
type(df)
#split the data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Text, df.Cause, test_size=0.33, random_state=12)

#Create dtm by using word occurence
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer( )
X_train_counts = count_vect.fit_transform(X_train)
X_train_counts.shape
count_vect.get_feature_names()

dtm1 = pd.DataFrame(X_train_counts.toarray().transpose(), index = count_vect.get_feature_names())
dtm1=dtm1.transpose()
dtm1.head()
dtm1.to_csv('dtm1.csv',sep=',')

#SVM
from sklearn.linear_model import SGDClassifier
import numpy as np
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer(use_idf=False)),
                      ('clf', SGDClassifier(
                                            alpha=1e-3
                                             ))
                    ])

text_clf.fit(X_train, y_train)  

predicted = text_clf.predict(X_test)
 
print(metrics.confusion_matrix(y_test, predicted))
print(np.mean(predicted == y_test) )


from sklearn.model_selection import GridSearchCV
# If we give this parameter a value of -1, 
#grid search will detect how many cores are installed and uses them all:
parameters = {
                  'tfidf__use_idf': (True, False),
                   'clf__alpha': (1e-2, 1e-3),
                }
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(X_train, y_train)
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
###################################################################
# Analyze the distribution
#words = [word for doc in cleaned for word in doc]
#type(cleaned)

#fd_words = FreqDist(words)
#fd_words.plot()
#fd_most_common=fd_cat.most_common(40)


 