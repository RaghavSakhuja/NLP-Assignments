class Tokenizer:
    def __init__(self):
        self.vocab = {}
        self.merges = {}
    
    def learn_vocabulary(self, corpus, num_merges):
        # Initialize vocabulary with characters

        self.vocab = {char: 1 for word in corpus for char in word}
        self.vocab['/'] = len(corpus)

        new_corpus = self.add_end_of_word_symbol(corpus)

        separated_corpus =self.separated_corpus(new_corpus)
        
        for merge in range(num_merges):
            # Calculate pair frequencies
            pairs = self.get_pair_frequencies(separated_corpus)
            
            if not pairs:
                break
            
            # Get the most frequent pair
            most_frequent_pair = max(pairs, key=pairs.get)
            
            # Merge the most frequent pair
            separated_corpus=self.merge_pair(most_frequent_pair,separated_corpus)
        
        # Generate vocabulary after merges
        self.generate_vocab_after_merges()
    
    def add_end_of_word_symbol(self, corpus):
        return [word + "/" for word in corpus]

    def separated_corpus(self, corpus):
        return [list(word) for word in corpus]

    def get_pair_frequencies(self, separated_corpus):
        pair_freq = {}
        for word in separated_corpus:
            pairs = [(word[i], word[i + 1]) for i in range(len(word) - 1)]
            for pair in pairs:
                pair_freq[pair] = pair_freq.get(pair, 0) + 1
        return pair_freq
    
    def merge_pair(self, pair,separated_corpus):
        # Update vocabulary after merging the pair

        self.vocab[''.join(pair)] = self.vocab[pair[0]] + self.vocab[pair[1]]
        # del self.vocab[pair[0]]
        # del self.vocab[pair[1]]

        # Update merges
        self.merges[pair] = self.vocab[''.join(pair)]
        
        #update separated corpus after merging the pair
        new_corpus = []
        for word in separated_corpus:
            new_word = []
            i=0
            while i < len(word) :
                if(i==len(word)-1):
                    new_word.append(word[i])
                    break
                if word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(''.join(pair))
                    i+=1
                else:
                    new_word.append(word[i])
                i+=1
            new_corpus.append(new_word)
        
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
corpus = ["low", "lower", "newest", "widest"]
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
