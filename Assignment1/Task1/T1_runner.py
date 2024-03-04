# Example Usage
from A1_T1_sub1_2021274_sub2_2021286 import Tokenizer

corpus = []
with open("corpus.txt", "r") as corpus_file:
    for line in corpus_file:
        words=line.split()
        for word in words:
            corpus.append(word)



num_merges = 100

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

# Getting the merge rules from the text file
merge_rules = []
with open("merge_rules.txt","r") as merge_rules_file:
    for rule in merge_rules_file:
        word_0,word_1 = rule.split(",")
        word_1 = word_1[:-1]
        merge_rules.append((word_0,word_1))

# Reading the test samples from the text file
test_samples =  []
with open("Test_Samples.txt","r") as test_file:
    for line in test_file:
        line = line[:-1]
        test_samples.append(line)

# Creating the tokens and appending to the tokenized samples file
with open("tokenized_samples.txt", "w") as tokenized_file:
    for sample in test_samples:
        sample_tokens = tokenizer.tokenize(sample,merge_rules)
        tokens = []
        for i in sample_tokens:
            for j in i:
                tokens.append(j)
        tokens_str = ",".join(tokens)
        tokenized_file.write(f"{tokens_str}\n")