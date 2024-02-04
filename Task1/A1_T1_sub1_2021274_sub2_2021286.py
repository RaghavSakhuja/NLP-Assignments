# A1_T1_sub1_2021274_sub2_2021286
import re


class Tokenizer:
    def __init__(self):
        self.vocab = set()
        self.merges = []
        self.eol='$'
    
    def learn_vocabulary(self, corpus, num_merges):
        # Initialize vocabulary with characters

        self.vocab = {char for word in corpus for char in word}
        self.vocab.add(self.eol)

        frequency_corpus = self.process_corpus(corpus)
        
        for i in range(num_merges):
            # Calculate pair frequencies
            pairs = self.get_pair_frequencies(frequency_corpus)
            
            if not pairs:
                break
            
            # Get the most frequent pair
            most_frequent_pair = max(pairs, key=pairs.get)
            
            # Merge the most frequent pair
            frequency_corpus=self.merge_pair(most_frequent_pair,frequency_corpus)
        
        # Generate vocabulary after merges
        # self.generate_vocab_after_merges()
    

    def process_corpus(self,corpus):
        result = []

        # Adding end-of-word symbol
        result = [word + self.eol for word in corpus]

        # Separating the corpus into lists of characters
        result = [list(word) for word in result]

        # Counting character frequencies
        freq = {}
        for word in result:
            freq[" ".join(word)] = freq.get(" ".join(word), 0) + 1

        return freq

    def get_pair_frequencies(self, frequencies_corpus):
        pair_freq = {}
        for word_str in frequencies_corpus:
            word = word_str.split()
            multiple=frequencies_corpus[word_str]
            for i in range(len(word)-1):
                pair = (word[i], word[i + 1])
                pair_freq[pair] = pair_freq.get(pair, 0) + multiple
        return pair_freq
    
    def merge_pair(self, pair,frequency_corpus):
        # Update vocabulary after merging the pair

        # self.vocab[''.join(pair)] = self.vocab[pair[0]] + self.vocab[pair[1]]
        # del self.vocab[pair[0]]
        # del self.vocab[pair[1]]
        self.vocab.add(''.join(pair))

        # Update merges
        self.merges.append(pair)
        
        #update frequency corpus after merging the pair
        new_corpus={}
        for word_str in frequency_corpus:
            word=word_str.split()
            multiple=frequency_corpus[word_str]
            new_word=[]
            i=0
            while i<len(word):
                if i<len(word)-1 and (word[i],word[i+1])==pair:
                    new_word.append("".join(pair))
                    i+=2
                else:
                    new_word.append(word[i])
                    i+=1
            new_corpus[" ".join(new_word)]=new_corpus.get(" ".join(new_word),0)+multiple

        
        return new_corpus
    
    def generate_vocab_after_merges(self):
        # Generate vocabulary after all merges
        final_vocab = {}
        for word in self.vocab:
            final_vocab[word] = self.vocab[word]
            for pair in self.merges:
                final_vocab[word.replace(''.join(pair), '')] = self.vocab[word]
        self.vocab = final_vocab

    def fold_case(self,sentence):
        return sentence.lower()
    
    def remove_punctuations(self,sentence):
        punctuation_removal_regex = r'[^\w\s]'
        sentence_punctuation_removed = re.sub(punctuation_removal_regex,'',sentence)
        sentence_punctuation_removed = sentence_punctuation_removed.strip()
        return sentence_punctuation_removed
    
    def get_character_list(self,sentence):
        word_list = sentence.split(" ")
        character_list=[]
        for word in word_list:
            lst=[]
            for char in word:
                lst.append(char)
            lst.append(self.eol)
            character_list.append(lst)
        return character_list
    
    # Function for tokenizing the string
    def tokenize(self, sentence,merge_rules):
        # Folding the case of the sentence
        sentence_case_folded = self.fold_case(sentence)

        # Removing punctuations and white spaces at the ends of the string
        sentence_punctuation_removed = self.remove_punctuations(sentence_case_folded)    
        # Converting string into a list of characters
        character_list = self.get_character_list(sentence_punctuation_removed)
        
        # Tokenizing the string
        finalTokenizedList=character_list.copy()
        for word_index in range(len(finalTokenizedList)):
            for rule in merge_rules:
                length = len(finalTokenizedList[word_index])
                i=0
                while(i<length-1):
                    if finalTokenizedList[word_index][i]==rule[0] and finalTokenizedList[word_index][i+1]==rule[1]:
                        mergedWord = [finalTokenizedList[word_index][i] + finalTokenizedList[word_index][i+1]]
                        word_before = finalTokenizedList[word_index][0:i]
                        word_after = finalTokenizedList[word_index][i+2:]
                        finalTokenizedList[word_index] = word_before + mergedWord + word_after
                        length -=1
                    i=i+1
        return  finalTokenizedList

