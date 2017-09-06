import pandas as pd
import gzip
import numpy as np
from surprise import SVD, Dataset
from surprise import evaluate, print_perf
from surprise.evaluate import GridSearch
def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')

df = getDF('ratings.gz')

alldata=df.as_matrix()
x=alldata.shape[0]
print alldata.shape
#print alldata[0]
alldata=np.random.permutation(alldata)
traindata =alldata[:x/4,[0,1,6,4]]
testdata =alldata[x/4+1:,[0,1,6,4]]
print ' AFTER PARSE'
print alldata
alldata=alldata[:,[0,1,6,4]]
df = pd.DataFrame(alldata)
df.to_csv("fulldata.csv")
'''
df = pd.DataFrame(traindata)
df.to_csv("train.csv")
df = pd.DataFrame(testdata)
df.to_csv("test.csv")
'''