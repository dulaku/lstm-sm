import os
import random
import string

#Run this on your line-separated raw data, then word2vec on the unified.txt file
# ./word2vec -train /path/to/train_data.txt -output /path/to/Data_Processed/vectordata.txt -window 10 -threads 24 -iter 10 -min-count 0 -size 300

texts = {"cache.txt"} 			# The names of the files containing the lines you want divided up among all your data sets.
test_reserved = {"augment.txt"}		# The names of the files, if any, containing lines you want reserved for your validation and test sets.
					# This is a dumb hack that ensures each has some relevant tokens, useful for testing purposes but probably not for real applications.
					# Note that each line in this file appears once in EACH of the validation and test sets.
sourcefolder = os.path.expanduser("./Dataset_Raw")	#Path to the files in texts and test_reserved
outfolder = os.path.expanduser("./Dataset_Processed")	#Path to the directory where the output files will be saved
if not os.path.isdir(outfolder):
	os.makedirs(outfolder) #Note: There's a race condition here; if you make the folder somehow between the if and this, things break.
setsizes = {'train':0.75, 'test':0.15}	# Proportion of lines that will wind up in each set. Validation set size is implicit, and there is no error-checking.
min_count = 64				#Minimum number of times an arbitrary token must appear in global vocab list to avoid being overwritten
min_hashtag = 1000			#Minimum number of times a hashtag must appear in global vocab list
min_user = 1000				#Minimum number of times a username must appear in global vocab list
max_special_density = 0.1		#Maximum proportion of a tweet that can be replaced due to failing one of the above requirements before the tweet is simply not added
min_len = 4				#Minimum number of tokens in a tweet before it will be dropped
max_len = 40				#Maximum number of tokens in a tweet before it will be dropped

global_set = set()
reserved_set = set()
reserved_set_filtered = set()
final_set = set()
train_set = set()
test_set = set()
val_set = set()
vocab_count = {}

punctuator = "" #Unnecessary globals is bad, so sue me.
for symbol in string.punctuation:
	if symbol != "@" and symbol != "#":
		punctuator += symbol #Don't strip @ (leave it to identify twitter users) or # (leave it to preserve hashtags)

translator = str.maketrans({key: None for key in punctuator})
digits_trans = str.maketrans({'0' : 'zero ', '1' : 'one ', '2' : 'two ', '3' : 'three ', '4' : 'four ', '5' : 'five ',
                              '6' : 'six ', '7' : 'seven ', '8' : 'eight ', '9' : 'nine '})
remove_specials = str.maketrans({'@': None, '#': None}) # Used to ensure that, if either of these characters do appear in a word, they're the only sorts of special characters that do.
linecount = 0
def procfile(text, outset):
	textfile = open(os.path.join(sourcefolder, text), 'r+')
	global linecount
	for line in textfile:
		linecount += 1
		if line == '':
			continue
		for char in line:
			if ord(char) > 128:
				continue # If a non-ASCII character exists in the tweet, skip it. Kind of a kludge; lots of assumptions about Unicode that are merely usually true.
		if line[:2].lower() == "rt": #Suppress retweets, but this way we keep ones whose original tweet wasn't present
			line = line[:2]
		towrite = ""
		splitline = line.split() # split line on whitespace
		has_words = False #initially assume line has no useful words
		for word in splitline:
			stripped = (word.translate(translator)).lower() # Strip all punctuation except @ and #, then force lowercase
			if stripped[:4] != "http": # Don't write hyperlinks
				if stripped.isalnum():  #Skip words that have non-alphanumeric characters that survived
					towrite += stripped.translate(digits_trans) + " " # Convert digits to words to reduce vocab size
					has_words = True # Hang on, couldn't I just check the length of towrite?
				elif '@' in stripped or '#' in stripped: #If allowed punctuation is in the words
					stripped_more = stripped.translate(remove_specials)
					if stripped_more.isalnum(): # If the above removal fixed it
						towrite += stripped.translate(digits_trans) + " " # The original stripped version is fine
						has_word = True
		if has_words:
			outset.add(towrite)
	textfile.close()

for text in texts:
	procfile(text, global_set)
for text in test_reserved:
	procfile(text, reserved_set)

print("Original Line Count", linecount)

for line in global_set:
	splitline = line.split() # split line on whitespace again for further processing
	for word in range(len(splitline)):
		try:
			vocab_count[splitline[word]] += 1
		except KeyError:
			vocab_count[splitline[word]] = 1

for line in reserved_set: #It was easier to copy paste this section rather than make it a function. Fix it one day.
	splitline = line.split() # split line on whitespace again for further processing
	for word in range(len(splitline)):
		try:
			vocab_count[splitline[word]] += 1
		except KeyError:
			vocab_count[splitline[word]] = 1

def setfilter(inset, outset):
	for line in inset:
		splitline = line.split()
		tosave = ""
		wordcount = 0
		specialcount = 0
		for word in range(len(splitline)):
			if '#' in splitline[word] and vocab_count[splitline[word]] < min_hashtag:
				splitline[word] = "<HASHTAG>"
				specialcount += 1
			elif '@' in splitline[word] and vocab_count[splitline[word]] < min_user:
				splitline[word] = "<USER>"
				specialcount += 1
			elif vocab_count[splitline[word]] < min_count:
				splitline[word] = "<OOV>"
				specialcount += 1
			wordcount += 1
			tosave += splitline[word] + " "
		if specialcount/wordcount < max_special_density and wordcount >= min_len and wordcount <= max_len: # Don't save tweets that are more than 30% replacements or that are only one word long
			outset.add(tosave)

setfilter(global_set, final_set)
setfilter(reserved_set, reserved_set_filtered)
print("Reserved Set Size: ", len(reserved_set_filtered))

del global_set
del reserved_set

for line in final_set:
	assignment = random.random()
	if assignment < setsizes['train']:
		train_set.add('<s> ' + line + '</s>')
	elif assignment < setsizes['train'] + setsizes['test']:
		test_set.add('<s> ' + line + '</s>')
	else:
		val_set.add('<s> ' + line + '</s>')
for line in reserved_set_filtered:
	test_set.add('<s> ' + line + '</s>')
	val_set.add('<s> ' + line + '</s>')

del final_set
del reserved_set_filtered

train_set = list(train_set)
test_set = list(test_set)
val_set = list(val_set)

train_set.sort(key=lambda strng: -1*len(strng.split()))
test_set.sort(key=lambda strng: -1*len(strng.split()))
val_set.sort(key=lambda strng: -1*len(strng.split()))
vocab_count = {} #Reuse this to calculate word frequencies post-replacement IN TRAINING SET ONLY
token_count = 0 #Total count of all tokens in training set
linecount = 0
outext = open(os.path.join(outfolder, 'unified.txt'), 'w')
for line in train_set:
	linecount += 1
	outext.write(line)
	outext.write('\n')
print("Train Set Size", linecount)
linecount = 0
for line in test_set:
	linecount += 1
	outext.write(line)
	outext.write('\n')
print("Test Set Size", linecount)
linecount = 0
for line in val_set:
	linecount += 1
	outext.write(line)
	outext.write('\n')
print("Validation Set Size", linecount)
outext = open(os.path.join(outfolder, 'train_data.txt'), 'w')
for line in train_set:
	splitline = line.split()
	for token in splitline:
		token_count += 1
		try:
			vocab_count[token] += 1
		except KeyError:
			vocab_count[token] = 1
	outext.write(line)
	outext.write('\n')
outext.close()
outext = open(os.path.join(outfolder, 'test_data.txt'), 'w')
for line in test_set:
	outext.write(line)
	outext.write('\n')
outext.close()
outext = open(os.path.join(outfolder, 'val_data.txt'), 'w')
for line in val_set:
	outext.write(line)
	outext.write('\n')
outext.close()
outext = open(os.path.join(outfolder, 'freqs.txt'), 'w')
outext.write("<SUM> " + str(token_count) + '\n')
for token in vocab_count.keys():
	outext.write(token + ' ' + str(vocab_count[token]) + '\n')
outext.close()
