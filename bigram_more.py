import time
import pandas as pd
from transformers import pipeline
from itertools import combinations

classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True,device=0)

def emotion_scores(words):
    # print(words)
    emotions = classifier(words)
    # print(emotions[0])
    return [[e['score'] for e in emotions[i]] for i in range(len(emotions))]

def generate_bigrams(corpus):
    return list(combinations(corpus, 2))

start_time = time.time()

corpus = ""
with open("corpus.txt", "r") as corpus_file:
    for line in corpus_file:
        corpus += line + " "

corpus = corpus.split()
corpus= list(set([word.lower() for word in corpus]))

# Generate bigrams from the corpus
bigrams = generate_bigrams(corpus)
# print(len(bigrams))
# exit(0)

# Accumulate bigram data in a list
bigram_input = []
for bigram in bigrams:
#   # Combine bigram words into a single string
    for bigram2 in bigrams[bigram]:
        if(bigrams[bigram][bigram2]>=3):
            bigram_text = " ".join([bigram,bigram2])
            # Append bigram data to the list
            bigram_input.append(bigram_text)

#calculate emotions for all bigrams
emotions = emotion_scores(bigram_input)

# Process the bigram
bigram_data=[]
for i in range(len(bigram_input)):
    bigram_data.append([bigram_input[i]]+emotions[i])


# Create a DataFrame for all bigrams
columns = ["bigram", "sadness", "joy", "love", "anger", "fear", "surprise"]
df = pd.DataFrame(bigram_data, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv("emotion_scores_bigrams_more.csv", index=False)

end_time = time.time()
elapsed_time = end_time - start_time

# # Print the result
print(f"Elapsed time: {elapsed_time} seconds")