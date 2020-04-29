import random
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.optim as optim
import torch.nn as nn

from Utility.LanguageLoader import LanguageLoader
from Utility.EncoderRNN import EncoderRNN
from Utility.DecoderRNN import AttnDecoderRNN

from Utility.logger import Logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



MAX_LENGTH = 20
num_iterations = 200000
print_every = 1000

SOS_token = 0
EOS_token = 1

loader = LanguageLoader(MAX_LENGTH)
logger = Logger()

fig, ax = plt.subplots()

def asMinutes(seconds):
    minutes = math.floor(seconds / 60)
    seconds -= minutes * 60
    return '%dm %ds' % (minutes, seconds)

def timeSince(since, percent):
    now = time.time()
    elapsed_seconds = now - since
    es = elapsed_seconds / (percent)
    rs = es - elapsed_seconds
    return '%s (- %s)' % (asMinutes(elapsed_seconds), asMinutes(rs))

def indexesFromSentences(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentences(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for i in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0,0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[i])
            decoder_input = target_tensor[i]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for i in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[i])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def trainModels(encoder, decoder, num_iters, print_every=1000, learning_rate=0.01):
    print('training started...')
    start = time.time()
    switch_variable = 0
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(num_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, num_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / num_iters), iter, iter / num_iters * 100, print_loss_avg))
        
            plot_loss_avg = plot_loss_total / print_every
            plot_losses.append(plot_loss_avg)            
            plot_loss_total = 0
            logger.log(plot_losses, "NLP_Tutorial/Logging/", "losses_{0}.txt".format(switch_variable))                
            logger.log_txt(plot_losses, "NLP_Tutorial/Logging/", "losses_txt_{0}.txt".format(switch_variable))
            showPlot(plot_losses)

            logger.log_state_dict(encoder.state_dict(), "NLP_Tutorial/models/RNNEncoder_{0}".format(switch_variable))
            logger.log_state_dict(decoder.state_dict(), "NLP_Tutorial/models/AttnDecoder_{0}".format(switch_variable))
            switch_variable += 1
            switch_variable %= 2

def showPlot(points):    
    
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.pause(0.1)

if __name__ == '__main__': #son windows ding mit mehreren Subprozessen.....

    input_lang, output_lang, pairs = loader.prepareData('deutsch', 'english',True)
    print(random.choice(pairs))

    teacher_forcing_ratio = 0.5

    hidden_size = 256
    encoder = EncoderRNN(input_lang.num_words, hidden_size).to(device)   

    attn_decoder = AttnDecoderRNN(hidden_size, output_lang.num_words, dropout_p=0.1).to(device)

    
    trainModels(encoder, attn_decoder, num_iterations, print_every=print_every)

    ######################################################################
    #
    #evaluateRandomly(encoder, attn_decoder)
    logger.log_state_dict(encoder.state_dict(), "NLP_Tutorial/models/RNNEncoder")
    logger.log_state_dict(attn_decoder.state_dict(), "NLP_Tutorial/models/AttnDecoder")
    plt.show()
    

