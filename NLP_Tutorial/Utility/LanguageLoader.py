
from Lang import Lang

class LanguageLoader:
    def __init__(self, max_length=20):
        self.max_length = max_length

    def filterPairs(self, pairs):
        return [pair for pair in pairs if filterPair(pair)]

    def filterPair(self, pair):    
        return len(pair[0].split(' ')) < self.max_length and len(pair[1].split(' ')) < self.max_length

    def readLangs(self, lang1, lang2, reverse=False):
        print("Reading lines...")

        # Read the file and split into lines       
        lines = open('./NLP Tutorial/data/%s-%s_pairs.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
        pairs = [line.split('\t') for line in lines]
        # Split every line into pairs and normalize
    
        # reverse pairs, make Lang instanstances
        if reverse:
            pairs = [list(reversed(pair)) for pair in pairs]
            input_lang = Lang(lang2)
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)

        return input_lang, output_lang, pairs

    def prepareData(self, lang1, lang2, reverse=False):
        input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    
        print("Read %s sentence pairs" % len(pairs))
        pairs = filterPairs(pairs)
    
        print("Trimmed to %s sentence pairs" % len(pairs))
    
        print("Counting words...")
        for pair in pairs:
            input_lang.AddSentence(pair[0])
            output_lang.AddSentence(pair[1])
    
        print("Counted words:")
        print(input_lang.name, input_lang.num_words)
        print(output_lang.name, output_lang.num_words)
        return input_lang, output_lang, pairs
