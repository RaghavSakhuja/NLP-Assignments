import numpy as np
import pandas as pd
# import seaborn as sns
import random
import csv

class BigramLM:
    def __init__(self):
        self.bigrams={}
        self.uniquewords=[]
        self.co_matrix=[]
        self.unigrams={}
        self.totalcount=0
        self.bol="#"
        self.eol="$"
        self.bigramCount = 0

    def add_data(self,corpus):
        
        with open(corpus) as f:
            words = f.read().split()
        for i in range(len(words)-1):
            if words[i] not in  self.bigrams:
                self.bigrams[words[i]]={}
                self.bigramCount+=1
            if words[i+1] not in self.bigrams[words[i]]:
                self.bigrams[words[i]][words[i+1]]=0
                self.bigramCount+=1
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
        # print(df)
        # print(sns.heatmap(df))
        # df.to_csv("CSVs/NoSmoothing.csv")
        
    def Laplace_learn(self,k=1):
        l=[]
        for i in range(len(self.uniquewords)):
            x=[]
            
            
            for j in range(len(self.uniquewords)):
                if self.uniquewords[i] in self.bigrams:
                    if self.uniquewords[j] in self.bigrams[self.uniquewords[i]]:
                        numerator = self.bigrams[self.uniquewords[i]][self.uniquewords[j]] + k #Reference for the formula Notes and https://www.youtube.com/watch?v=gCI-ZC7irbY
                        denominator = self.unigrams[self.uniquewords[i]] + (k*len(self.uniquewords))
                        p = numerator/denominator
                        l.append((p*(self.unigrams[self.uniquewords[i]]/self.totalcount),self.uniquewords[i],self.uniquewords[j]))
                        x.append(p)
                        # print("yes")
                    
                    else:
                        p = k / (self.unigrams[self.uniquewords[i]] + (k * len(self.uniquewords)))
                        x.append(p)
                        l.append((p*(self.unigrams[self.uniquewords[i]]/self.totalcount),self.uniquewords[i],self.uniquewords[j]))
                else:
                    p = k / (k * len(self.uniquewords))
                    x.append(p)
                    l.append((p*(self.unigrams[self.uniquewords[i]]/self.totalcount),self.uniquewords[i],self.uniquewords[j]))
            self.co_matrix.append(x)
        df=pd.DataFrame(self.co_matrix,columns=self.uniquewords,index=self.uniquewords)
        l.sort(reverse=True)
        print(l[:5])
        # print(df)
        # print(sns.heatmap(df))
        # df.to_csv("CSVs/Laplace.csv")
        

        
    def KneserNey_learn(self, d=0.75):
        for i in range(len(self.uniquewords)):
            x = []

            for j in range(len(self.uniquewords)):
                if self.uniquewords[i] in self.bigrams:
                    if self.uniquewords[j] in self.bigrams[self.uniquewords[i]]:
                        numerator = max(self.bigrams[self.uniquewords[i]][self.uniquewords[j]] - d, 0)
                        denominator = self.unigrams[self.uniquewords[i]]
                        continuation_count = sum(1 for k in self.bigrams if self.uniquewords[j] in self.bigrams[k])
                        continuation_prob = continuation_count / len(self.bigrams)

                        w = sum(1 for k in self.bigrams[self.uniquewords[i]] if self.bigrams[self.uniquewords[i]][k] > 0)
                        alpha = (d / denominator) * w

                        kneser_ney_prob = numerator / denominator + alpha * continuation_prob
                        x.append(kneser_ney_prob)
                    else:
                        x.append(0)
                else:
                    x.append(0)
            self.co_matrix.append(x)
        df = pd.DataFrame(self.co_matrix, columns=self.uniquewords, index=self.uniquewords)
        # print(df)
        # sns.heatmap(df)
        # df.to_csv("CSVs/KneserNey.csv")


    
    def generate_sentences(self,sentence_length):
        # r=random.choice(self.uniquewords)
        r="i"
        s=r+" "
        k=self.uniquewords.index(r)
        for i in range(sentence_length-1):
            w=random.choices(self.uniquewords,weights=self.co_matrix[k],k=1)[0]
            s+=w+" "
            k=self.uniquewords.index(w)
        return s
    
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
    
    def find_top5_bigrams(self, output_file="top5prob.csv"):
        l = []

        with open(output_file, "w", newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Bigram', 'Count'])  # Write header to CS
            
            for i in self.bigrams:
                for j in self.bigrams[i]:
                    l.append((self.bigrams[i][j]/self.totalcount, i + " " + j))
            l.sort(reverse=True)
            top5 = l[:5]

            for count, bigram in top5:
                csv_writer.writerow([bigram, count])
                
    def update_corpus(self,corpus):
        with open("corpus2.txt","w") as f1:
            with open(corpus,"r") as f2:
                sentences=f2.readlines()
                for i in sentences:
                    i=i.strip()
                    f1.write("# "+i+" $\n")
                    
    def p_first(self,corpus):
            with open(corpus) as f:
                words = f.read().split()
                d={}
                total=0
                for i in range(len(words)):
                    if (words[i]==self.bol):
                        if(words[i+1] in d):
                            d[words[i+1]]+=1
                        else:
                            d[words[i+1]]=1
                        total+=1
                for i in d:
                    d[i] = d[i]/total
                d2={'word':d.keys(),'prob':d.values()}
                df=pd.DataFrame(d2)
                df.to_csv("first_prob.csv")
                
                


b1=BigramLM()
b1.add_data("corpus.txt")
# print(b1.bigrams)
# b1.learn()
# b1.KneserNey_learn()
# b1.Laplace_learn()
# b1.find_top5_bigrams()
# b1.update_corpus("corpus.txt")
# b1.p_first("corpus2.txt")
# print(b1.findsentenceprob("i stand here i feel empty a class post count link href http mooshilu"))

#Create csvs for kneser and laplace and add kneser ney freq calculation