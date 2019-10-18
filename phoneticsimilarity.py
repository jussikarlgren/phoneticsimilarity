import pickle
from logger import logger
import sklearn.metrics
from scipy.spatial.distance import cosine
import numpy
import random
from nltk import word_tokenize

# loglevels for logger
debug = True
error = True
monitor = True



# facts about phonetics
vowels = "aeiouäöåüy"  # english will have some trouble with y but never mind, hey
consonants = "bcdgfhjklmnpqrstvwxz"
alphabet = vowels + consonants
# we do not now have bigram phonemes (ch, tj, and diphtongs etc) and we should but hey
# each articulatory class has a random vector
articulatory_classes = {}
articulatory_classes["klusil"] = "ptkqbdg"
articulatory_classes["frikativ"] = "szjfvrl"  # not quite true but hey
articulatory_classes["nasal"] = "mn"
articulatory_classes["labial"] = "pbmw"
articulatory_classes["palatal"] = "j"
articulatory_classes["dental"] = "tdsnl"
articulatory_classes["velar"] = "kghq"
articulatory_classes["likvid"] = "lrw"
articulatory_classes["tonlös"] = "ptkqfs"
articulatory_classes["tonande"] = "bdgvrlmnzw"
articulatory_classes["aspirerad"] = "pqtkh"
articulatory_classes["front"] = "ieyüäö"
articulatory_classes["back"] = "uoåa"
articulatory_classes["closed"] = "iyuü"
articulatory_classes["mid"] = "eöoå"
articulatory_classes["open"] = "äa"
articulatory_classes["rounded"] = "yuüåoö"
articulatory_classes["unrounded"] = "ieäa"
articulatory_classes["vowel"] = vowels  # fake to weight this higher


# init the vector space
dimensionality = 200
sparseness = 200
itemtable = {}
permutationtable = {}


# vector space predicates
def newrandomvector(dimensionality:int, nonzeroelements:int):
    vec = numpy.zeros(dimensionality)
    nonzeros = random.sample(range(dimensionality), nonzeroelements)
    flip = nonzeroelements // 2
    blip = 1
    for nz in nonzeros:
        vec[nz] = blip
        flip -= 1
        if flip <= 0:
            blip = -1
    return vec


def normalise(vector:numpy.array):
    return vector  # to be written


def createpermutation(k:list):
    permutation = random.sample(range(k), k)
    return permutation


def permute(vector:numpy.array, permutation:list):
    newvector = numpy.zeros(dimensionality)
    for i in range(len(permutation)):
        newvector[permutation[i]] = vector[i]
    return newvector


# provide base for representation
# this representation should be pickled and saved if a representation is intended to be reused
def generate_alphabet():
    for c in alphabet:
        vector = newrandomvector(dimensionality, sparseness)
        for a in articulatory_classes:
            if a not in itemtable:
                itemtable[a] = newrandomvector(dimensionality, sparseness)
            if c in articulatory_classes[a]:
                vector = vector + itemtable[a]
        vector = normalise(vector)
        itemtable[c] = vector
        permutationtable[c] = createpermutation(dimensionality)
    dist = sklearn.metrics.pairwise_distances(list(itemtable.values()), metric="cosine")
    logger(dist, debug)
    for i in itemtable:
        for j in itemtable:
            logger("{}\t{}\t{}".format(i, j, 1-round(cosine(itemtable[i], itemtable[j]), 3)), debug)
    itemtable["dummyvector"] = newrandomvector(dimensionality, dimensionality)
    permutationtable["vowelsequence"] = createpermutation(dimensionality)


# simple word reader
def doonerawtextfile(filename:str):
    vocabulary = set()
    with open(filename, errors="replace", encoding='utf-8') as inputtextfile:
        for textline in inputtextfile:
            words = word_tokenize(textline.lower())
            vocabulary.update(words)
    return vocabulary


# saving and retrieving a model
def outputphoneticmodel(filename:str):
    try:
        representation = {}
        representation["vectors"] = itemtable
        representation["permutations"] = permutationtable
        with open(filename, 'wb') as outfile:
            pickle.dump(representation, outfile)
    except IOError:
            logger("Could not write >>" + filename + "<<", error)


def inputphoneticmodel(vectorfile:str):
    global itemtable, permutationtable
    try:
        cannedmodel = open(vectorfile, 'rb')
        representation = pickle.load(cannedmodel)
        itemtable = representation["vectors"]
        itemtable = representation["permutations"]
    except IOError:
        logger("Could not read from >>" + vectorfile + "<<", error)


# string processing
def windows(sequence:str, window:int):
    windowlist = []
    if window > 0:
        windowlist = [sequence[ii:ii + window] for ii in range(len(sequence) - window + 1)]
    return windowlist

def sequencepermutationvector(sequence:str, v:numpy.array):
    for c in sequence:
        try:
            v = permute(v, permutationtable[c])
        except KeyError:
            pass
    return v


def sequenceadditivevector(sequence:str, permutation:list):
    v = numpy.zeros(dimensionality)
    for c in sequence:
        try:
            v = itemtable[c] + permute(v, permutation)
        except KeyError:
            pass
    return v



# hypothesis: "bag of characters", baseline --- containing the same characters is an indication of closeness
test_character_presence = False

# hypothesis: vowels are more important than other items for token similarity
test_vowel_sequence = True

# hypothesis: bigrams and trigrams of characters are useful
test_ngrams = True

# hypothesis: characters in the beginning of the token are more important that later ones
# if this constant is set to zero all characters are equal, if more than zero, weight tapers off towards end of token
# decremented at each vowel encountered (approximating syllables)
test_descent = 0.3

# hypothesis: some suffixes can be disregarded entirely
test_skip_suffixes = True
skippables = ["ing", "ed"]

def process(token:str):
    global descent, itemtable, permutationtable
    vector = numpy.zeros(dimensionality)
    vowelsequence = ""
    if test_skip_suffixes and token.endswith(tuple(skippables)):  # double test not to have iterate over skippables
        for tc in skippables:
            if token.endswith(tc):
                token = token[:-len(tc)]
                break
    for c in token:
        if test_character_presence:
            try:
                vector = vector + itemtable[c] * (1 - test_descent) ** len(vowelsequence)
            except KeyError:
                pass
        if test_vowel_sequence:
            if c in vowels:
                vowelsequence = vowelsequence + c
    bigrams = windows(token, 2)
    trigrams = windows(token, 3)
    if test_ngrams:
        for ngram in bigrams + trigrams:
            v = sequencepermutationvector(ngram, itemtable["dummyvector"])
            vector = vector + v
    if test_vowel_sequence:
        vector = vector + sequenceadditivevector(vowelsequence, permutationtable["vowelsequence"])
    return vector


# run one actual experiment
filename="/home/jussi/data/alice_adventures.txt"
generate_alphabet()
vocabulary = doonerawtextfile(filename)
vectorspace = {}
for token in vocabulary:
    vectorspace[token] = process(token)

neighbours = 10
for token in vectorspace:
    neighbourliness = {}
    for anothertoken in vectorspace:
        neighbourliness[anothertoken] = 1-round(cosine(vectorspace[token],vectorspace[anothertoken]),10)
    neighbourhood = sorted(neighbourliness,key=lambda k: neighbourliness[k], reverse=True)[:neighbours]
    for n in neighbourhood:
        logger("{}\t{}\t{}".format(token,n,neighbourliness[n]), debug)




