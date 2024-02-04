from q2_1_CsvAdded import BigramLM


b1=BigramLM()
b1.add_data("corpus.txt")

b1.learn()
b1.KneserNey_learn()
b1.Laplace_learn()
b1.find_top5_bigrams()