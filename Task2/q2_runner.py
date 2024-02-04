from A1_T2_Q1_2021258_Q2_2021555 import BigramLM


b1=BigramLM()
b1.add_data("corpus.txt")

b1.learn()
b1.KneserNey_learn()
b1.Laplace_learn()
b1.find_top5_bigrams()