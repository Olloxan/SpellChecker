path = './books/Project_Gutenberg.rtf'
shakespeare = open(path).read()
training_text = shakespeare.split('\nTHE END', 1)[-1]

print(len(training_text))

chars = list(sorted(set(training_text)))
chars_to_idx = {ch: idx for idx, ch in enumerate(chars)}

print(len(chars))

#def char_rnn_model(num_chars, num_layers, num_nodes=512, dropout=0.1):
