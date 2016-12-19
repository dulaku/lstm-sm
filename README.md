# lstm-sm
A model for token salience, based on an LSTM language model

# Introduction

Processing large volumes of mostly-irrelevant data to find the pieces that are useful for a particular task is still a challenge. This model is meant to help with keyword searches by expanding a short list of known keywords to a larger list by adding words that appear in similar contexts. The program will produce a list of suggested words; it's still up to a human to read the list and decide which suggestions are worthwhile.

This is a pretty niche use, but it can also serve as an example of a medium-sized machine learning project implemented in Torch - it's got a few interesting bells and whistles that might be interesting to people coping with limited hardware, including a dynamic-length sequence dataloader and a sequence shuffler that respects bucketing. Most significantly, it grabs the states of two LSTM-based language models (not just their outputs) as they process sequences, and uses that as input to a feedforward classifier that determines whether the next-predicted-word should be in the set of known keywords.

This was written to get me a degree - JJ_CC.pdf is the paper I wrote describing it and its motivation, and goes into far more depth on the background, design, and "why it's interesting" of this thing.

## Requirements:
&nbsp;&nbsp;[word2vec](https://github.com/dav/word2vec)  
&nbsp;&nbsp;[Torch](http://torch.ch/)  
&nbsp;&nbsp;&nbsp;&nbsp;[cunn](https://github.com/torch/cunn) -- If using CUDA  
&nbsp;&nbsp;&nbsp;&nbsp;[rnn](https://github.com/Element-Research/rnn)  
&nbsp;&nbsp;&nbsp;&nbsp;[dataload](https://github.com/Element-Research/dataload)  
&nbsp;&nbsp;&nbsp;&nbsp;[optim](https://github.com/torch/optim)  
&nbsp;&nbsp;&nbsp;&nbsp;[lfs](https://keplerproject.github.io/luafilesystem/)  
&nbsp;&nbsp;&nbsp;&nbsp;[tds](https://github.com/torch/tds)  
&nbsp;&nbsp;For practical purposes, you will want a CUDA-capable GPU, but it is not strictly necessary.

##Overview

The general pipeline for use is:

1. Obtain a dataset, in the form of a text file. Put independent sequences on separate lines; what that means depends on your data. This tool was designed to handle Tweets, with each Tweet being its own sequence.
2. Preprocess the data to simplify later steps, using preprocessor.py
3. Run w2v on the train_data.txt file produced in Step 2. 
4. Run lstm-sm.lua with your preferred arguments.
5. Read testpositives.txt

## Detailed Instructions

###preprocessor.py
preprocessor.py doesn't accept command line arguments. Fixing this is on the to-do list. For now, you will generally need to edit the source code -- the relevant lines are listed below, and commented with their meanings in the code.

1. You should edit lines 8, 9, 10, and 11 to contain filenames and paths according to your needs. By default, input files are assumed to be in a directory called Dataset_Raw, and output files are saved to a directory called Dataset_Processed, both located in the same directory as preprocessor.py
2. There are a number of parameters that also can be set to affect the preprocessing behavior on lines 17 through 22; the defaults replace 
   any token that appears less than *64* times, 
   any Twitter username that appears less than *1000* times, and
   any hashtag that appears less than *1000* times
  with generic tokens, and drop any resulting Tweets that are 
   more than *10%* replaced in this way, 
   shorter than *4* tokens, or 
   longer than *40* tokens
3. Additionally, the script will convert all tokens to lowercase, all digits to their word equivalents (1 -> "one", 12 -> "one two", etc), remove punctuation other than # and @, remove "rt" if they are the first two characters in a tweet, and drop tokens that start with "http:" or contain non-ASCII characters. After this, duplicates are removed.
4. The preprocessor prints some basic information about the dataset when it completes, such as the number of lines in each file. This isn't saved anywhere, so copy it if you want it.
    
###word2vec
An example command for folks who aren't familiar with the tool (for further detail, look into the linked page in Requirements):  
`./word2vec -train /path/to/train_data.txt -output /path/to/Data_Processed/vectordata.txt -window 10 -threads 24 -iter 10 -min-count 0 -size 300`

###lstm-sm.lua
Command-line arguments are documented in the code, and can be enumerated by running  
`th ./lstm-sm.lua --help`  
In particular, arguments that are lists of strings (such as targwords, the list of known keywords) should be specified as "{'string'}", as in  
`th ./lstm-sm.lua --targwords "{'obama'}"`  
Without any arguments specifying otherwise, the program will save everything it needs somewhere in its current directory; caches of the dataset will be saved in a directory Dataset_Processed/cache/ located within the same directory as the file, and everything else will be saved directly in the same directory.

Questions, comments, suggestions, and corrections are welcome at jasonj[address delimiter]iastate[decimal point]edu

