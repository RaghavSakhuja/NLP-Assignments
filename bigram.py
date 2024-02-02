import time
import pandas as pd
from transformers import pipeline
# from transformers.utils import logging

# logging.set_verbosity_info()
# logger = logging.get_logger("transformers")
# logger.info("INFO")

classifier = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion',batch_size=80, return_all_scores=True,device=0)

def emotion_scores(words):
    emotions = classifier(words)
    return [[e['score'] for e in emotions[i]] for i in range(len(emotions))]

start_time = time.time()
corpus="corpus.txt"
bigrams={}
with open(corpus) as f:
    words = f.read().split()
    for i in range(len(words)-1):
        if words[i] not in  bigrams:
            bigrams[words[i]]={}
        if words[i+1] not in bigrams[words[i]]:
            bigrams[words[i]][words[i+1]]=0
        bigrams[words[i]][words[i+1]]+=1
    
# Accumulate bigram data in a list
bigram_input = []
for bigram in bigrams:
#     # Combine bigram words into a single string
    for bigram2 in bigrams[bigram]:
        # if(bigrams[bigram][bigram2]>=3):
            bigram_text = " ".join([bigram,bigram2])
            # Append bigram data to the list
            bigram_input.append(bigram_text)

#calculate emotions for all bigrams

emotions = emotion_scores(bigram_input)
print(len(emotions))
# Process the bigram
bigram_data=[]
for i in range(len(bigram_input)):
    bigram_data.append([bigram_input[i]]+emotions[i])


# Create a DataFrame for all bigrams
columns = ["bigram", "sadness", "joy", "love", "anger", "fear", "surprise"]
df = pd.DataFrame(bigram_data, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv("emotion_scores_bigrams_exis.csv", index=False)

end_time = time.time()
elapsed_time = end_time - start_time

# # Print the result
print(f"Elapsed time: {elapsed_time} seconds")