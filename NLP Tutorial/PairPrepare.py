from io import open
import re
import unicodedata

lang1 = "deutsch"
lang2 = "english"

def unicodeToASCII(string):
    return ''.join(char for char in unicodedata.normalize('NFD', string) if unicodedata.category(char) != 'Mn')

def normalizeString(string):
    string = unicodeToASCII(string.lower().strip())
    string = re.sub(r"([.!?])", r" \1", string)
    string = re.sub(r"[^a-zA-Z.!?]+", r" ", string)
    return string

# Read the file and split into lines   
print("opening Languagefile...")

# Read the file and split into lines   
lines = open('./NLP Tutorial/data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
print("creating pairs")
# Split every line into pairs and normalize
pairs = [[normalizeString(string) for string in line.split('\t')[:2]] for line in lines]

languagePairFile = open('./NLP Tutorial/data/%s-%s_pairs.txt' % (lang1, lang2), 'w+', encoding='utf-8')
print("saving pairs")

for pair in pairs:    
    languagePairFile.write('%s\t%s\n' % (pair[0], pair[1])) 

languagePairFile.close()

print("creation complete")