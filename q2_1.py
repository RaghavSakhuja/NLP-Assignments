import numpy as np
import pandas as pd

class BigramLM:
    def __init__(self):
        self.bigrams={}
        self.uniquewords=[]
        self.co_matrix=[]
        self.unigrams={}
        self.totalcount=0

    def add_data(self,corpus):
        with open(corpus) as f:
            words = f.read().split()
        for i in range(len(words)-1):
            if words[i] not in  self.bigrams:
                self.bigrams[words[i]]={}
            if words[i+1] not in self.bigrams[words[i]]:
                self.bigrams[words[i]][words[i+1]]=0
            self.bigrams[words[i]][words[i+1]]+=1
            if words[i] not in self.unigrams:
                self.unigrams[words[i]]=1
            else:
                self.unigrams[words[i]]+=1
            self.totalcount+=1
        
        if words[-1] not in self.unigrams:
                self.unigrams[words[-1]]=1
        else:
            self.unigrams[words[-1]]+=1
        self.totalcount+=1
        self.uniquewords=list(self.unigrams.keys())
    
    def learn(self):
        for i in range(len(self.uniquewords)):
            x=[]
            
            for j in range(len(self.uniquewords)):
                if self.uniquewords[i] in self.bigrams:
                    if self.uniquewords[j] in self.bigrams[self.uniquewords[i]]:
                        x.append(self.bigrams[self.uniquewords[i]][self.uniquewords[j]]/self.unigrams[self.uniquewords[i]])
                        # x.append(self.bigrams[self.uniquewords[i]][self.uniquewords[j]])
                    else:
                        x.append(0)
                else:
                    x.append(0)
            self.co_matrix.append(x)
        df=pd.DataFrame(self.co_matrix,columns=self.uniquewords,index=self.uniquewords)
        print(df)
        
        # print(self.unigrams)
        # print(self.uniquewords)
        # for i in self.co_matrix:
        #     print(i)
        # print("yes")
    
    def generate_sentences(self):
        pass
    
    def findsentenceprob(self,sentence):
        p=1
        words=sentence.strip(".").split()
        if len(words)>0:
            p*=(self.unigrams[words[0]] if words[0] in self.unigrams else 0) 
            for i in range(len(words)-1):
                idx1=(self.uniquewords.index(words[i]) if words[i] in self.uniquewords else -1)
                idx2=(self.uniquewords.index(words[i+1]) if words[i+1] in self.uniquewords else -1)
                if(idx1!=-1 and idx2!=-1):
                    x=self.co_matrix[idx1][idx2]
                    # print(x)
                    p*=x
                else:
                    p*=0
                    break
        else:
            p=0
        # print(p)
        return p

b1=BigramLM()
b1.add_data("corpus.txt")
b1.learn()
print(b1.findsentenceprob("i stand here i feel empty a class post count link href http mooshilu"))