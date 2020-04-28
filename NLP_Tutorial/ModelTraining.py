
from Utility.LanguageLoader import LanguageLoader


MAX_LENGTH = 20
num_iters = 1

loader = LanguageLoader(MAX_LENGTH)




if __name__ == '__main__': #son windows ding mit mehreren Subprozessen.....

    input_lang, output_lang, pairs = loader.prepareData('deutsch', 'english',True)
    print(random.choice(pairs))

    teacher_forcing_ratio = 0.5

    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.num_words, hidden_size).to(device)   

    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.num_words, dropout_p=0.1).to(device)

    
    trainIters(encoder1, attn_decoder1, num_iters, print_every=5000)

    ######################################################################
    #
    encoder1_state_dict = encoder1.state_dict()
    torch.save(encoder1_state_dict, './NLP Tutorial/models/RNNEncoder')

    attn_decoder1_state_dict = attn_decoder1.state_dict()
    torch.save(encoder1_state_dict, './NLP Tutorial/models/AttnDecoder')
    evaluateRandomly(encoder1, attn_decoder1)

