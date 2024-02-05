import numpy as np
import pandas as pd
# import seaborn as sns
import random
import csv

class BigramLM:
    def __init__(self):
        self.bigrams={}
        self.uniquewords=[]
        self.co_matrix_no=[]
        self.co_matrix_laplace=[]
        self.co_matrix_kneser=[]
        self.unigrams={}
        self.totalcount=0
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
            self.co_matrix_no.append(x)
        df=pd.DataFrame(self.co_matrix_no,columns=self.uniquewords,index=self.uniquewords)
        df.to_csv('CSVs/No_Smoothing.csv')

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
            self.co_matrix_laplace.append(x)
        
        df=pd.DataFrame(self.co_matrix_laplace,columns=self.uniquewords,index=self.uniquewords)
        df.to_csv('CSVs/Laplace_Smoothing.csv')
        
        l.sort(reverse=True)
        with open("Top5Bigrams/top5Laplace.csv", "w", newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Probability', 'Word1', 'Word2'])  # Write header to CSV
            for p, word1, word2 in l[:5]:
                csv_writer.writerow([p, word1, word2])
        
    def KneserNey_learn(self, d=0.75):
        l=[]
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
                        
                        l.append((kneser_ney_prob*(self.unigrams[self.uniquewords[i]]/self.totalcount),self.uniquewords[i],self.uniquewords[j]))
                        x.append(kneser_ney_prob)
                    else:
                        l.append((0,self.uniquewords[i],self.uniquewords[j]))
                        x.append(0)
                else:
                    l.append((0,self.uniquewords[i],self.uniquewords[j]))
                    x.append(0)
            self.co_matrix_kneser.append(x)
        df = pd.DataFrame(self.co_matrix_kneser, columns=self.uniquewords, index=self.uniquewords)
        df.to_csv('CSVs/KneserNey_Smoothing.csv')

        l.sort(reverse=True) 
        with open("Top5Bigrams/top5Kneser.csv", "w", newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Probability', 'Word1', 'Word2'])  # Write header to CSV

            for p, word1, word2 in l[:5]:
                csv_writer.writerow([p, word1, word2])
    
    def findsentenceprob(self,sentence,smooth):
        if smooth=='No':
            co_matrix=self.co_matrix_no
        elif smooth=='Laplace':
            co_matrix=self.co_matrix_laplace
        elif smooth=='Kneser':
            co_matrix=self.co_matrix_kneser
        l=[]
        p=1
        words=sentence.strip(".").split()
        if len(words)>0:
            p*=(self.unigrams[words[0]] if words[0] in self.unigrams else 0) 
            for i in range(len(words)-1):
                idx1=(self.uniquewords.index(words[i]) if words[i] in self.uniquewords else -1)
                idx2=(self.uniquewords.index(words[i+1]) if words[i+1] in self.uniquewords else -1)
                if(idx1!=-1 and idx2!=-1):
                    x=co_matrix[idx1][idx2]
                    # print(x)
                    p*=x
                    l.append(x)
                else:
                    p*=0
                    l.append(0)
                    break
        else:
            p=0
        return (p,l)
    
    def find_top5_bigrams(self, output_file="Top5Bigrams/top5prob.csv"):
        l = []
        with open(output_file, "w", newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['Bigram', 'Probability'])  # Write header to CS
            for i in self.bigrams:
                for j in self.bigrams[i]:
                    l.append((self.bigrams[i][j]/self.totalcount, i + " " + j))
            l.sort(reverse=True)
            top5 = l[:5]
            for count, bigram in top5:
                csv_writer.writerow([bigram, count])
              

