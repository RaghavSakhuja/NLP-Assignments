from collections import Counter

class Tokenizer:
    def __init__(self):
        self.vocab = set()
        self.merges = []
        self.eol='/'
    
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
        result = [word + "/" for word in corpus]

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
    
    def tokenize(self, sample):
        tokens = []
        for word in sample.split():
            tokens.extend(self.split_word(word))
        return tokens
    
    def split_word(self, word):
        # Split a word based on the learned rules
        for pair in self.merges:
            if ''.join(pair) in word:
                return word.split(''.join(pair))
        return [word]

# Example Usage
corpus = ["low", "lower", "newest", "widest","low","new","wider"]
num_merges = 10

tokenizer = Tokenizer()
tokenizer.learn_vocabulary(corpus, num_merges)

# Print vocabulary
with open("vocabulary.txt", "w") as vocab_file:
    for token in tokenizer.vocab:
        vocab_file.write(token + "\n")
        print(list(token))

# Print merge rules
with open("merge_rules.txt", "w") as merge_file:
    for pair in tokenizer.merges:
        merge_file.write(f"{pair[0]},{pair[1]}\n")

# Tokenize test samples
test_samples = ["lowest", "newer", "wider"]
with open("tokenized_samples.txt", "w") as tokenized_file:
    for sample in test_samples:
        tokens = tokenizer.tokenize(sample)
        tokenized_file.write(",".join(tokens) + "\n")
