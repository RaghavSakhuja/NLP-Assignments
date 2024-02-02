import time
import pandas as pd
from transformers import pipeline
from itertools import combinations

classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)

def emotion_scores(words):
    emotions = classifier(words)
    return [e['score'] for e in emotions[0]]

def generate_bigrams(corpus):
    return list(combinations(corpus, 2))

start_time = time.time()

corpus = ""
with open("data/corpus.txt", "r") as corpus_file:
    for line in corpus_file:
        corpus += line + " "

corpus = corpus.split()
corpus= list(set([word.lower() for word in corpus]))
# corpus = corpus[:10]

# Generate bigrams from the corpus
bigrams = generate_bigrams(corpus)

# Accumulate bigram data in a list
bigram_data = []
for bigram in bigrams:
    # Combine bigram words into a single string
    bigram_text = " ".join(bigram)
    
    # Process the bigram
    emotions = emotion_scores(bigram_text)
    
    # Append bigram data to the list
    bigram_data.append([bigram_text] + emotions)

# Create a DataFrame for all bigrams
columns = ["bigram", "sadness", "joy", "love", "anger", "fear", "surprise"]
df = pd.DataFrame(bigram_data, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv("emotion_scores_bigrams.csv", index=False)

end_time = time.time()
elapsed_time = end_time - start_time

# Print the result
print(f"Elapsed time: {elapsed_time} seconds")
