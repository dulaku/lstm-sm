local active_history = {} -- TODO: Make this part of the neural net object instead of a global variable (read: bad, but functional, hack)
-- Alternatively, see if the information isn't already being squirreled away somewhere to support backpropagation-through-time

require 'rnn'
require 'optim'
require 'lfs'

local dl = require 'dataload' -- provides a handy table-printing function
local tds = require 'tds'

math.randomseed(413612) --Initialize RNG seed to get consistent results across tests. Arbitrary number is arbitrary.

local version = 0.1 --Not necessary, but nice to know.

--[[
Begin command line arguments
--]]

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Language Model and then a Salience Model based on it.')
cmd:text('Example:')
cmd:text('th lmsm.lua --cuda --device 1 --id j_random_model --batchsize 64')
cmd:text('Options:')

-- General
cmd:option('--cuda', true, 'Whether or not to use CUDA.')
cmd:option('--device', 1, 'CUDA device to be used. No effect if cuda is false.')
cmd:option('--savepath', lfs.currentdir() .. '/', 'Path to directory where files will be saved. If necessary, logs will also be loaded from this folder.')
cmd:option('--id', 'Testing', 'String identifying the dataset. Doesn\'t do much yet.')
cmd:option('--samplemax', 16, 'Maximum length of a sample language sequence generated to illustrate language model performance at each validation improvement. -1 means no limit (WARNING: Often produces infinite loops, especially in early epochs). Regardless, sample terminates at end-of-sentence tag </s> or the start-of-sentence tag <s>.')
cmd:option('--seedwords', '{"<s>", "claro"}', 'A sequence of words used to seed the language models for sample sentence generation. If not provided or invalid, {"<s>"} is used instead. Since the same sequence gets fed into the forward and backward readers, you probably will only get good sampling results from one model.') -- TODO: separate sequences for forward and backward reader
cmd:option('--losshistfile', 'losshist.csv', 'Filename for saving the loss histories of all models. One history per line, in the order: forward training loss, forward validation loss, backward training loss, backward validation loss, salience training loss, salience validation lost. Loss values are a sum of all losses across the epoch; it may be more helpful to divide by the number of sequences to get loss per sequence.')
cmd:option('--posfile', 'testpositives.txt', 'Filename that will be used to save "<token>: <count>" pairs of tokens predicted to be in targwords.')
cmd:option('--t_printfreq', 100, 'Number of batches to train on before printing a status update. A number less than 1 suppresses status updates.')
cmd:option('--v_printfreq', 100, 'Number of sequences to validate before printing a status update. A number less than 1 suppresses status updates.')

-- Training
cmd:option('--train_lm', true, 'Whether to train the language model or not. If false, just runs a test set. If you don\'t load a pre-trained model in this case, the result will (almost) certainly be nonsense.')
cmd:option('--train_sm', true, 'Whether to train the salience model or not. If false, just runs a test set.')
cmd:option('--lm_lr', 0.0003, 'Initial learning rate for language models.') -- Karpathy constant
cmd:option('--sm_lr', 0.0001, 'Initial learning rate for salience model.')
cmd:option('--lm_minlr', 0.00001, 'Minimum learning rate for language models.')
cmd:option('--sm_minlr', 0.00001, 'Minimum learning rate for salience model.')
cmd:option('--lm_lrdecay', 1, 'Factor by which learning rate will be multiplied each epoch. Language model.')
cmd:option('--sm_lrdecay', 1, 'Salience model learning rate decay.')
cmd:option('--lm_momentum', 0.9, 'Momentum factor for training updates; higher momentum gives more weight to past updates than current updates. Must be between 0 and 1. Language.')
cmd:option('--sm_momentum', 0.9, 'Momentum factor for salience model.')
cmd:option('--lm_weightdecay', 0.0001, 'A factor that penalizes large weights in the network. Keeps them within a range that tends to be better for learning. Language model.')
cmd:option('--sm_weightdecay', 0.0001, 'Weight decay for salience model.')
cmd:option('--maxepoch', 1024, 'Maximum number of epochs to train for each model.')
cmd:option('--earlystop', 20, 'Number of epochs to wait without an improvement on validation data before stopping. Each model ends its own training in this way independently.')
cmd:option('--silent', false, 'Don\'t print anything to stdout.')

-- Models
cmd:option('--loadlm', '', 'If not the empty string, load a language model from the given path instead of making one. Only the model is loaded; the optimizer state and secondary modules are recreated from scratch.')
cmd:option('--loadsm', '', 'If not the empty string, load a salience model classifier from the given path instead of making one. Only the model is loaded; the optimizer state and secondary modules are recreated from scratch.')
cmd:option('--uniform', -1, 'Initialize parameters using uniform distribution between -uniform and uniform. -1 means a module\'s default. Applies to all models.')
cmd:option('--lm_inputsize', 300, 'Size of language model input vectors. Incorrect values are not supported.') --TODO: Get from vectors file automatically.
cmd:option('--lm_hiddensize', '{200, 200, 200, 200, 200}', 'Number of units in each hidden layer of the language model. When more than one is in the list, each represents a layer in a stack.')
cmd:option('--sm_hiddensize', '{25, 25, 25}', 'Number of units in each hidden layer of the salience model. When more than one is in the list, each represents a layer in a stack.')
cmd:option('--lm_dropout', 0.1, 'Apply dropout with this probability after each LSTM layer in the language model. Values of 0 or less, or greater than  or equal to 1, disable dropout.')
cmd:option('--sm_dropout', 0.1, 'Apply dropout with this probability after each ReLU layer in the salience model.')

-- Data
cmd:option('--reppath', 'Dataset_Processed/vectordata.txt', 'Path, starting at savepath, to text file containing vector representations for the vocabulary.')
--cmd:option('--freqpath', 'Dataset_Processed/freqs.txt', 'Path to a text file containing the frequencies of each token in the training set.')
cmd:option('--targwords', '{"claro"}', 'Words the salience model will learn as significant.')
cmd:option('--dataset', 'Dataset_Processed/', 'Path within savepath to directory containing texts, in files named test_data.txt, train_data.txt, and val_data.txt')
cmd:option('--datacache', 'Dataset_Processed/cache/', 'Path to directory containing cached, preprocessed batches. These should still be shuffled before use.')
cmd:option('--batchnum', 1000, 'Maximum number of batches to load from disk into memory at a given time. 0 means the entire dataset.')
cmd:option('--batchsize', 3, 'Number of sequences to train on per batch.')

cmd:text()

local opt = cmd:parse(arg or {})
opt.lm_hiddensize = loadstring(" return "..opt.lm_hiddensize)()
opt.sm_hiddensize = loadstring(" return "..opt.sm_hiddensize)()
opt.seedwords =     loadstring(" return "..opt.seedwords)()
opt.targwords =     loadstring(" return "..opt.targwords)()
for _,word in ipairs(opt.targwords) do
  print(word)
end

opt.lm_inputsize = opt.lm_inputsize == -1 and opt.lm_hiddensize[1] or opt.lm_inputsize

opt.sm_inputsize = 0 -- Determine size of input vector to significance model
for _,j in ipairs(opt.lm_hiddensize) do
  opt.sm_inputsize = opt.sm_inputsize + 2*j -- by summing the lm hidden layer sizes
end 
--Multiply by 2 because there is a forward and a backward model, but they're identical
--A change to allow different architectures for each of those will require changing this to add their layers individually

if not opt.silent then
  table.print(opt)
end

if opt.cuda then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.device)
end

opt.sal_positives = {}
--A table whose entries are <token>:<count>, where <count> is the number of times that <token> was predicted to be in
--the target set for the salience model.

--[[
END command line arguments
--]]

--[[
BEGIN Main definition
--]]
function main()
  opt.embeds, opt.vocab, opt.ivocab = get_embeddings()
  local fwd, bwd, sm = {}, {}, {}
  fwd.model, fwd.targmod, fwd.crit, fwd.optim = lm_build('fwd') -- Reads from sequence start to end
  bwd.model, bwd.targmod, bwd.crit, bwd.optim = lm_build('bwd') -- Reads from sequence end to start
  sm.model, sm.targmod, sm.crit, sm.optim = sm_build('sal')
  local log = log_init()
  local lm_stop_count = 0 -- Count of epochs in which neither the forward nor backward reader advanced beyond its best result
  local sm_stop_count = 0 -- Ditto for the salience model

  -- Finish opt setup below. This is done in main() because it calls for functions defined below, so bad organization was inevitable.
  -- TODO: Move the util functions into their own file, then you can move the next few lines up with their siblings outside main()

  opt.trainsize, opt.validsize, opt.testsize = get_line_nums() -- placeholder variables for dataset sizes; further processing necessary.
  opt.trainsize = math.floor(opt.trainsize / opt.batchsize) -- After getting raw number of examples from the line count or user, convert trainsize to the number of batches to be used; this is a more useful number. Note floor; extra sequences are discarded pre-shuffle, so they'll never get used.
  if not opt.silent then
    print("Data split into:\n\tTraining set of ", opt.trainsize, " batches of size ", opt.batchsize, " sequences.\n\tValidation set of ", opt.validsize, " sequences\n\tTest set of ", opt.testsize, " sequences")
  end

  if opt.train_lm then
    for i=1,opt.maxepoch do
      opt.lm_epoch = i
      if not opt.silent then
        print("\n Language Model Epoch: ".. opt.lm_epoch)
      end
      local lm_fwd_loss,lm_bwd_loss = lm_train(fwd, bwd)
      table.insert(log.fw_trainloss, lm_fwd_loss)
      table.insert(log.bw_trainloss, lm_bwd_loss)
      if not opt.silent then
        print("Current Forward LM Train Loss: " .. lm_fwd_loss ..
            "\nCurrent Backward LM Train Loss: " .. lm_bwd_loss)
      end
      lm_fwd_loss,lm_bwd_loss = lm_valtst(fwd, bwd, 'val_data.txt') -- Having trained, determine model goodness with validation set, not training
      table.insert(log.fw_valloss, lm_fwd_loss)
      table.insert(log.bw_valloss, lm_bwd_loss)
      if not opt.silent then
        print("Current Forward LM Validation Loss: " .. lm_fwd_loss ..
             " Best Forward LM Prior Loss: " .. log.fw_minvalloss .. 
            "\nCurrent Backward LM Validation Loss: " .. lm_bwd_loss ..
             " Best Backward LM Prior Loss: " .. log.bw_minvalloss)
      end
      if lm_fwd_loss >= log.fw_minvalloss and lm_bwd_loss >= log.bw_minvalloss then
        lm_stop_count = lm_stop_count + 1
      else
        lm_stop_count = 0
      end

      if lm_fwd_loss < log.fw_minvalloss then -- Separate if statements because generally one does not imply the other
        log.fw_minvalloss = lm_fwd_loss
        prog_save('fwd', fwd, log)
      end

      if lm_bwd_loss < log.bw_minvalloss then
        log.bw_minvalloss = lm_bwd_loss
        prog_save('bwd', bwd, log)
      end    

      if lm_stop_count >= opt.earlystop then
        break
      end

      if not opt.silent then
        print("Last Improvement Epoch", i - lm_stop_count)
      end
      
      local lr_cand = fwd.optim.learningRate * opt.lm_lrdecay -- lr_candidate
      fwd.optim.learningRate = math.max(lr_cand, opt.lm_minlr)
      lr_cand = bwd.optim.learningRate*opt.lm_lrdecay
      bwd.optim.learningRate = math.max(lr_cand, opt.lm_minlr)

      if not opt.silent then
        print("Forward sample:")
        lm_sample(fwd.model) -- No sense in sampling if we never print it.
        print("Backward sample: (read backward)") -- TODO: More sensible sampling output
        lm_sample(bwd.model)
      end
    end

    fwd.model = torch.load(opt.savepath .. 'fwd.t7')--Restore best models
    bwd.model = torch.load(opt.savepath .. 'bwd.t7')
    
    if opt.cuda then
      fwd.model:cuda()
      bwd.model:cuda()
    end
  end
  
  if not opt.silent then
    print("\nBest Forward Model Sample:")
    lm_sample(fwd.model)
    print("\nBest Backward Model Sample: (read backward)")
    lm_sample(bwd.model)
    print() -- Just a newline
  end

  if opt.train_sm then
    for j=1,opt.maxepoch do
      opt.sm_epoch = j
      if not opt.silent then print("\n Salience Model Epoch: " .. opt.sm_epoch) end
      opt.sal_positives = {}
      local sm_loss = sm_train(sm, fwd, bwd)
      table.insert(log.sm_trainloss, sm_loss)
      if not opt.silent then print("Current SM Train Loss: " .. sm_loss) end
      opt.sal_positives = {}
      sm_loss = sm_valtst(sm, fwd, bwd, 'val_data.txt')
      table.insert(log.sm_valloss, sm_loss)
      if not opt.silent then 
        print("Current SM Validation Loss: " .. sm_loss ..
             " Best SM Prior Loss: ".. log.sm_minvalloss)
      end
      if sm_loss >= log.sm_minvalloss then
        sm_stop_count = sm_stop_count + 1
      else
        log.sm_minvalloss = sm_loss
        prog_save('sal', sm, log)
        sm_stop_count = 0
      end
      if sm_stop_count >= opt.earlystop then
        break
      end
      if not opt.silent then print("Last Improvement Epoch", j - sm_stop_count) end
      local lr_cand = sm.optim.learningRate * opt.sm_lrdecay -- lr_candidate
      sm.optim.learningRate = math.max(lr_cand, opt.sm_minlr)
    end

    sm.model = torch.load(opt.savepath .. 'sal.t7')
    
    if opt.cuda then
      sm.model:cuda()
    end
  end
  
  if not opt.silent then print("Beginning tests...\n") end
  if not opt.silent then print("Language model tests...") end
  local lm_fwd_loss,lm_bwd_loss = lm_valtst(fwd, bwd, 'test_data.txt')
  if not opt.silent then
    print("Forward LM Test Loss: " .. lm_fwd_loss ..
        "\nBackward LM Test Loss: " .. lm_bwd_loss .. "\n")
  end
  if not opt.silent then print("Salience model tests...")end
  opt.sal_positives = {}
  local sm_loss = sm_valtst(sm, fwd, bwd, 'test_data.txt')
  if not opt.silent then print("SM Test Loss: " .. sm_loss) end
  
  prog_save('sal', sm, log) -- Overwrite final validation epoch salience predictions with test salience predictions and, in the likely event the final epoch wasn't saved, update the loss histories and such.
end

--[[
BEGIN Utility functions; TODO: Put these in another file and require that.
--]]

function nn.SeqLSTM:updateOutput(input) --Modified update version to store activations
  self.recompute_backward = true
  local c0, h0, x = self:_prepare_size(input)
  local N, T = x:size(2), x:size(1)
  self.hiddensize = self.hiddensize or self.outputsize -- backwards compat
  local H, R, D = self.hiddensize, self.outputsize, self.inputsize

  self._output = self._output or self.weight.new()

  -- remember previous state?
  local remember
  if self.train ~= false then -- training
    if self._remember == 'both' or self._remember == 'train' then
      remember = true
    elseif self._remember == 'neither' or self._remember == 'eval' then
      remember = false
    end
  else -- evaluate
    if self._remember == 'both' or self._remember == 'eval' then
      remember = true
    elseif self._remember == 'neither' or self._remember == 'train' then
      remember = false
    end
  end

  self._return_grad_c0 = (c0 ~= nil)
  self._return_grad_h0 = (h0 ~= nil)
  if not c0 then
    c0 = self.c0
    if self.userPrevCell then
      local prev_N = self.userPrevCell:size(1)
      assert(prev_N == N, 'batch sizes must be consistent with userPrevCell')
      c0:resizeAs(self.userPrevCell):copy(self.userPrevCell)
    elseif c0:nElement() == 0 or not remember then
      c0:resize(N, H):zero()
    elseif remember then
      local prev_T, prev_N = self.cell:size(1), self.cell:size(2)
      assert(prev_N == N, 'batch sizes must be constant to remember states')
      c0:copy(self.cell[prev_T])
    end
  end
  if not h0 then
    h0 = self.h0
    if self.userPrevOutput then
      local prev_N = self.userPrevOutput:size(1)
      assert(prev_N == N, 'batch sizes must be consistent with userPrevOutput')
      h0:resizeAs(self.userPrevOutput):copy(self.userPrevOutput)
    elseif h0:nElement() == 0 or not remember then
      h0:resize(N, R):zero()
    elseif remember then
      local prev_T, prev_N = self._output:size(1), self._output:size(2)
      assert(prev_N == N, 'batch sizes must be the same to remember states')
      h0:copy(self._output[prev_T])
    end
  end

  local bias_expand = self.bias:view(1, 4 * H):expand(N, 4 * H)
  local Wx = self.weight:narrow(1,1,D)
  local Wh = self.weight:narrow(1,D+1,R)

  local h, c = self._output, self.cell
  h:resize(T, N, R):zero()
  c:resize(T, N, H):zero()
  local prev_h, prev_c = h0, c0
  self.gates:resize(T, N, 4 * H):zero()
  if self.record then                                 -- Added this
    table.insert(active_history[#active_history], {}) -- through
  end                                                 -- this for salience model
  for t = 1, T do
    local cur_x = x[t]
    self.next_h = h[t]
    local next_c = c[t]
    local cur_gates = self.gates[t]
    cur_gates:addmm(bias_expand, cur_x, Wx)
    cur_gates:addmm(prev_h, Wh)
    cur_gates[{{}, {1, 3 * H}}]:sigmoid()
    cur_gates[{{}, {3 * H + 1, 4 * H}}]:tanh()
    local i = cur_gates[{{}, {1, H}}] -- input gate
    local f = cur_gates[{{}, {H + 1, 2 * H}}] -- forget gate
    local o = cur_gates[{{}, {2 * H + 1, 3 * H}}] -- output gate
    local g = cur_gates[{{}, {3 * H + 1, 4 * H}}] -- input transform
    self.next_h:cmul(i, g)
    next_c:cmul(f, prev_c):add(self.next_h)
    self.next_h:tanh(next_c):cmul(o)

    -- for LSTMP
    self:adapter(t)

    if self.maskzero then
      -- build mask from input
      local vectorDim = cur_x:dim() 
      self._zeroMask = self._zeroMask or cur_x.new()
      self._zeroMask:norm(cur_x, 2, vectorDim)
      self.zeroMask = self.zeroMask or ((torch.type(cur_x) == 'torch.CudaTensor') and torch.CudaByteTensor() or torch.ByteTensor())
      self._zeroMask.eq(self.zeroMask, self._zeroMask, 0)     
      -- zero masked output
      self:recursiveMask({self.next_h, next_c, cur_gates}, self.zeroMask)
    end
    if self.record then                                                             -- And also this
      active_history[#active_history][#active_history[#active_history]][t] = next_c -- through this
    end                                                                             -- for salience model
    prev_h, prev_c = self.next_h, next_c
  end
  self.userPrevOutput = nil
  self.userPrevCell = nil

  if self.batchfirst then
    self.output = self._output:transpose(1,2) -- T x N -> N X T
  else
    self.output = self._output
  end

  return self.output
end

function set_record(model) -- TODO: Make this a method when you remove the active_history variable.
  for _,lstm in pairs(model:findModules('nn.SeqLSTM')) do
    lstm.record = true
  end
end

function unset_record(model) -- TODO: See set_record() note.
  for _,lstm in pairs(model:findModules('nn.SeqLSTM')) do
    lstm.record = false
  end
end
function string:split(sSeparator, nMax, bRegexp)
  -- Python-like string splitting - see:
  -- http://lua-users.org/wiki/SplitJoin
  -- "true Python semantics for split"
  assert(sSeparator ~= '')
  assert(nMax == nil or nMax >= 1)

  local aRecord = {}

  if self:len() > 0 then
    local bPlain = not bRegexp
    nMax = nMax or -1

    local nField, nStart = 1, 1
    local nFirst,nLast = self:find(sSeparator, nStart, bPlain)
    while nFirst and nMax ~= 0 do
      aRecord[nField] = self:sub(nStart, nFirst-1)
      nField = nField+1
      nStart = nLast+1
      nFirst,nLast = self:find(sSeparator, nStart, bPlain)
      nMax = nMax-1
    end
    aRecord[nField] = self:sub(nStart)
  end
  return aRecord
end
function get_embeddings()
  --Returns word embeddings, vocab (table of Token:Index pairs), and
  --ivocab (table of Index:Token) pairs
  local vec_length = opt.lm_inputsize
  local embeds_path = opt.savepath .. opt.reppath
  --local freqs_path = opt.savepath .. opt.freqpath --Unused for now, but likely to be useful in a later version

  if io.open(embeds_path) == nil then
    error("Embeddings file unopenable!")
--  elseif io.open(freqs_path) == nil then
--    error("Frequencies file unopenable!")
  elseif type(vec_length) ~= "number" then
    error("Embedding vector size required.")
  end

  local embeds = tds.Hash({})
  local vocab = tds.Hash({}) -- Token:Index pairs
  local ivocab = tds.Vec({}) -- Index:Token pairs

  embeds["<null>"] = torch.Tensor(vec_length):zero() -- Null token to avoid needing to write special handling instructions for 0 padding in other code
  ivocab:insert("<null>")                            -- TODO: Evaluate code to see if there are any cases where this could ever actually be important.
  vocab["<null>"] = #ivocab                          --       If not, delete this so that errors happen if a null token winds up somewhere it shouldn't.

  for line in io.lines(embeds_path) do
    local r_split = string.split(line, " ")
    if not (vocab[r_split[1]]) then --If word not in vocabulary, which should always be true
      ivocab:insert(r_split[1]) -- Add word to end of ivocab
      vocab[r_split[1]] = #ivocab -- Record its ivocab index in vocab
    end
    embeds[r_split[1]] = torch.Tensor(vec_length):zero()
    for val=2, vec_length+1 do
      embeds[r_split[1]][val-1] = r_split[val]
    end
  end
  
  if embeds["<OOV>"] == nil then --<OOV> token needs to exist for words that didn't exist in the training set, so hardcode an arbitrary stand-in; there is no research suggesting this is wise. TODO: Implement statistically-sound representation; look at Markov Chains?
    embeds["<OOV>"] = torch.Tensor(vec_length):zero() -- Although this is identical to null, no null token ever gets used in training, so this shouldn't cause confusion.
    ivocab:insert("<OOV>")
    vocab["<OOV>"] = #ivocab
  end

  --local freqs = torch.Tensor(#vocab):type('torch.DoubleTensor'):fill(1) -- Word:Frequency Count, initialized to 1 so that multiplication does nothing to slots that aren't filled for whatever reason (for example, "<null>")

--  local countsum = 0
--  for line in io.lines(freqs_path) do
--    local split = string.split(line, " ")
--    if split[1] == "<SUM>" then
--      countsum = split[2]
--    else
--      freqs[vocab[split[1]]] = split[2]
--    end
--  end

--  local targfreq = 0
--  for _,token in ipairs(opt.targwords) do
--    if vocab[token] ~= nil then
--      targfreq = targfreq + freqs[vocab[token]]
--    end
--  end

--  local normfactor = torch.median(freqs)
--  freqs = torch.cdiv(torch.Tensor(#vocab):fill(countsum), freqs)
--  freqs:div(normfactor[1]) --normfactor is a 1D tensor, we need its sole element

--  targfreq = torch.Tensor({countsum/(targfreq*normfactor[1]),
--                           countsum/((countsum-targfreq)*normfactor[1])})

  if not opt.silent then
    print("Vocabulary size: ", #vocab)
  end

  return embeds, vocab, ivocab--, freqs, targfreq
end

function get_line_nums()
  --Returns the number of lines in the training data file, validation data file, and test data file, respectively
  local train, val, test = 0, 0, 0
  for _ in io.lines(opt.savepath .. opt.dataset .. 'train_data.txt') do
    train = train + 1
  end
  for _ in io.lines(opt.savepath .. opt.dataset .. 'val_data.txt') do
    val = val + 1
  end
  for _ in io.lines(opt.savepath .. opt.dataset .. 'test_data.txt') do
    test = test + 1
  end
  return train, val, test
end

function log_init()
  local log
  if (opt.loadlm ~= '' or opt.loadsm ~= '') and opt.savepath ~= '' then
    log = torch.load(opt.savepath .. "log.t7")
    if opt.loadlm == '' then
      --One-line assignments. See below, starting at "log.opt = opt" for a clearer version of the same
      log.fw_trainloss, log.bw_trainloss, log.fw_valloss, log.bw_valloss, log.fw_minvalloss, log.bw_minvalloss, log.lm_epoch, log.fwd_best_epoch, log.bwd_best_epoch = {}, {}, {}, {}, math.huge, math.huge, 0, 0, 0
    end
    if opt.loadsm == '' then
      log.sm_trainloss, log.sm_valloss, log.sm_minvalloss, log.sm_epoch, log.sm_best_epoch = {}, {}, math.huge, 0, 0
    end
    return log
  else
    log = {}
  end
  log.opt = opt
  log.dataset = opt.id
  log.fw_trainloss, log.bw_trainloss, log.sm_trainloss = {}, {}, {}
  log.fw_valloss,   log.bw_valloss,   log.sm_valloss   = {}, {}, {}
  log.fw_minvalloss, log.bw_minvalloss, log.sm_minvalloss = math.huge, math.huge, math.huge
  log.lm_epoch, log.fwd_best_epoch, log.bwd_best_epoch = 0, 0, 0
  log.sm_epoch, log.sm_best_epoch = 0, 0

  return log
end

function prog_save(modeltype, modeltable, log)
  if not opt.silent then print("Saving log...") end
  torch.save(opt.savepath .."log.t7" , log)
  torch.save(opt.savepath .. modeltype .. ".t7", modeltable.model:float()) -- Convert to floating point for use with non-CUDA systems.
  if opt.cuda then
    modeltable.model:cuda() -- the float() method changed it in-place, so fix it.
  end

  if modeltype == 'sal' then -- Nothing to write if the only model that needed saving was a language model
    local positives = io.open(opt.savepath .. "testpositives.txt", 'w')
    for pred, count in pairs(opt.sal_positives) do
      positives:write(pred .. ": " .. count .. "\n")
    end
    positives:close()
  end

  local hist = io.open(opt.savepath .. "losshist.csv", 'w')
  for _,history in ipairs({log.fw_trainloss, log.fw_valloss, log.bw_trainloss, log.bw_valloss, log.sm_trainloss, log.sm_valloss}) do
    for _,loss in ipairs(history) do
      hist:write(tostring(loss) .. ',')
    end
    hist:write('\n')
  end
  hist:close()

  if not opt.silent then print("Save complete!")end
end

function batches_load(targ_file, batchsize, batchnum, start) 
  --To keep a reasonable memory footprint, only loads a chunk of raw data at a time
  --One epoch of training takes several of these
  --targ_file - the file containing one-sequence-per-line raw data
  --batchsize - the size of batches to be returned by the iterators generated
  --chunksize - the maximum number of lines of data to be loaded
  --seq_start - the cursor position in targ_file to start at
  
  --TODO: Consider switching to a proper database instead of reading a text file in segments?
  batchsize = batchsize or opt.batchsize
  batchnum = batchnum or opt.batchnum
  
  if lfs.attributes(opt.savepath .. opt.datacache) == nil then
    lfs.mkdir(opt.savepath .. opt.datacache)
  end
    

  local cachepath = opt.savepath .. opt.datacache .. targ_file .. "_" .. tostring(batchsize) .. "_" .. tostring(batchnum) .. "_" .. tostring(start) .. '.t7'
  local cachestart_path = opt.savepath .. opt.datacache .. targ_file .. "_" .. tostring(batchsize) .. "_" .. tostring(batchnum) .. "_" .. tostring(start) .. 'S.t7'

  local cache = io.open(cachepath, "r") --TODO: Replace this method of checking for the cache file now that I'm using lfs.

  if cache ~=  nil then
    io.close(cache)
    cache = torch.load(cachepath)
    local cachestart = torch.load(cachestart_path)
    return cache, cachestart
  end

  local start_at
  local path = opt.savepath .. opt.dataset .. targ_file
  local file = io.open(path)
  file:seek("set", start)
  local batches = {}

  for k=1,batchnum do
    local tokens_forward = {}
    local tokens_backward = {}
    local seqlens = {}
    local paddings = {}
    local max_seqlen = 0

    for i=1,batchsize do
      local line = file:read()
      if line then
        table.insert(tokens_forward, {})
        table.insert(tokens_backward, {})
        local splitline = string.split(line, " ")
        seqlens[i] = #splitline -1 -- Each sequence contains n tokens, but each tensor only needs n-1 of them, and that's what we'll use this for
        if #splitline > max_seqlen then
          max_seqlen = #splitline
        end
        for i=1,#splitline do
          if(splitline[i]) then
            table.insert(tokens_forward[#tokens_forward], splitline[i])
            table.insert(tokens_backward[#tokens_backward], splitline[#splitline + 1 - i])
          end
        end
        start_at = file:seek()
      else
        start_at = 0
        break
      end
    end

    if(#tokens_forward < batchsize or #tokens_backward < batchsize) then -- The break was triggered above because we ran out of lines
      torch.save(cachepath, batches)
      torch.save(cachestart_path, 0)
      return batches, start_at
    end

    max_seqlen = max_seqlen - 1 -- Each sequence contains n elements, but each tensor contains n-1
    --The forward reader doesn't have an input corresponding to the last word or a target corresponding to the first
    --Vice versa for the backward reader

    local fwd_tens_in = torch.Tensor(max_seqlen, batchsize, opt.lm_inputsize):type('torch.DoubleTensor'):zero()
    local fwd_tens_tg = torch.Tensor(max_seqlen, batchsize):type('torch.DoubleTensor'):zero() -- Targets are 2D because the criterion is designed to take an index, not a one-hot representation
    local bwd_tens_in = torch.Tensor(max_seqlen, batchsize, opt.lm_inputsize):type('torch.DoubleTensor'):zero()
    local bwd_tens_tg = torch.Tensor(max_seqlen, batchsize):type('torch.DoubleTensor'):zero()

    for i=1,batchsize do
      paddings[i] = max_seqlen - seqlens[i]
      for j=1,max_seqlen do
        if tokens_forward[i][j+1] ~= nil then
          if opt.vocab[tokens_forward[i][j+1]] then
            fwd_tens_tg[j+paddings[i]][i] = opt.vocab[tokens_forward[i][j+1]]   
            -- For, say, a list that looks like [fill, my, box], in a batch with max_seqlen 4
            -- target looks like [0, 0, my_i, box_i]
            -- where _i indicates the index in the vocab for the word and 0 is empty padding
          else  --Else OOV token
            fwd_tens_tg[j+paddings[i]][i] = opt.vocab["<OOV>"]
          end

          if opt.vocab[tokens_forward[i][j]] then
            fwd_tens_in[j+paddings[i]][i] = opt.embeds[tokens_forward[i][j]]
            -- but input looks like [<null>_v, <null>_v, fill_v, my_v]
            -- where _v indicates the corresponding vector representation and <null>_v is all 0s
          else
            fwd_tens_in[j+paddings[i]][i] = opt.embeds["<OOV>"]
            -- Insert an <oov> character for unknown tokens
          end
        end
      end
    end

    for i=1,batchsize do
      for j=2,max_seqlen+1 do
        if tokens_backward[i][j] then --Need nested ifs; do nothing if this fails
          if opt.vocab[tokens_backward[i][j]] ~= nil then --Else OOV token
            bwd_tens_tg[j+paddings[i]-1][i] = opt.vocab[tokens_backward[i][j]]  
            -- See the forward block, above, for a helpful comment explaining what's going on here
            -- The difference is that the index j+paddings[i]-1 is the correct index here
          else
            bwd_tens_tg[j+paddings[i]-1][i] = opt.vocab["<OOV>"]
          end

          if opt.vocab[tokens_backward[i][j-1]] ~= nil then
            bwd_tens_in[j+paddings[i]-1][i] = opt.embeds[tokens_backward[i][j-1]] 
            
          else
            bwd_tens_in[j+paddings[i]-1][i] = opt.embeds["<OOV>"]
          end
        end
      end
    end
    table.insert(batches, {fwd_in = fwd_tens_in, fwd_targ = fwd_tens_tg,
        bwd_in = bwd_tens_in, bwd_targ = bwd_tens_tg, pads=paddings})
  end

  torch.save(cachepath, batches)
  torch.save(cachestart_path, start_at)

  return batches, start_at --Return table of batches and a cursor position for starting the next read
end

function batches_perm(batches)
  --Takes a table filled with data batches of format:
  --{{fwd_in, fwd_targ, bfwd_in, bwd_targ}}
  --Returns a similarly-formatted table with sequences shuffled
  --A given sequence can only be swapped with another sequence of the same tensor length,
  --in order to preserve the processing-speed benefits of bucketing.
  --Shuffling does not respect batch boundaries, which is actually the point - to get multiple
  --different batches from the same dataset.
  --Splits batches into logical decks of cards, with each card representing a sequence
  --This seemed like a good idea at the time.
  local decksize = 0
  for batch=1,#batches do
    decksize = decksize + 1 -- number of batches in this deck
    --If this is the last batch or the next batch has a different sequence length
    if batch == #batches or ((batches[batch+1].fwd_in):size(1) ~= (batches[batch].fwd_in):size(1)) then
      for item=(decksize*opt.batchsize),2,-1 do--Fisher-Yates shuffle
        local b_offset = batch-math.floor((item-1) / opt.batchsize)
        local s_offset = item % opt.batchsize+1
        local swapper = math.random(item-1)
        local b_swpset = batch-math.floor(swapper / opt.batchsize)
        local s_swpset = swapper % opt.batchsize+1
        
          batches[b_offset]["fwd_in"][{{}, s_offset}], 
          batches[b_swpset]["fwd_in"][{{}, s_swpset}] 
        = batches[b_swpset]["fwd_in"][{{}, s_swpset}]:clone(), -- clone() because Torch's memory-saving
          batches[b_offset]["fwd_in"][{{}, s_offset}]:clone()  -- design would result in all of these just

          batches[b_offset]["fwd_targ"][{{}, s_offset}],
          batches[b_swpset]["fwd_targ"][{{}, s_swpset}]
        = batches[b_swpset]["fwd_targ"][{{}, s_swpset}]:clone(), -- pointing at the same memory locations
          batches[b_offset]["fwd_targ"][{{}, s_offset}]:clone()  -- instead of being swapped

          batches[b_offset]["bwd_in"][{{}, s_offset}], 
          batches[b_swpset]["bwd_in"][{{}, s_swpset}]
        = batches[b_swpset]["bwd_in"][{{}, s_swpset}]:clone(), -- Also, this should probably be a utility
          batches[b_offset]["bwd_in"][{{}, s_offset}]:clone()  -- function. Could really save some space.

          batches[b_offset]["bwd_targ"][{{}, s_offset}], 
          batches[b_swpset]["bwd_targ"][{{}, s_swpset}]
        = batches[b_swpset]["bwd_targ"][{{}, s_swpset}]:clone(), -- Let's call it a TODO:
          batches[b_offset]["bwd_targ"][{{}, s_offset}]:clone()

          batches[b_offset]["pads"][s_offset], batches[b_swpset]["pads"][s_swpset]
        = batches[b_swpset]["pads"][s_swpset], batches[b_offset]["pads"][s_offset]
      end
      decksize = 0 --Reset decksize
    end
  end
  return batches
end
function sm_process_input(sm_input, fw_trget, pads)
  --sm_input[direction][layer][timestep][batch][neuron]
  -- TAKES a <number of models> x <number of layers> x <number of timesteps> table.
  -- <number of models> is always 2 - forward-reader and backward-reader, in that order.
  -- Layers run from the network's input to its output.
  -- Each of the tables corresponding to a timestep contains a <batch size> x <number of neurons in this layer> tensor of neuron states.
  -- Here, it's the hidden states (the value passed to the next layer and to the next timestep)
  
  -- TAKES a table of the target outputs for the forward-reader language model (an arbitrary choice; could've been backward, too).
  -- TAKES a table indicating how much left-padding each sequence in each batch got.
  
  -- RETURNS a <number of timesteps> x <batch size> x <number of neurons in all language models> tensor
  -- of inputs for the SM
  
  -- RETURNS a <number of timesteps> x <batch size> tensor of targets for the SM. Like the language
  -- model, the targets represent the indices of a one-hot vector, because that's what the criterion
  -- expects.
  
  -- TODO: See if the cell state of the LSTM works any better.
  -- TODO: Visualize the weights the first layer of the SM learns, see which segments of the LM are
  --       most useful. Hypothesis: With a lousy language model, it'll be the layers closest to the input.

  local seqlen = #sm_input[1][1]-1 -- -1 because each tensor has a spare <s> or </s> token that is uninteresting and won't be analyzed.
  local batchsize = fw_trget:size(2)
  --It's either opt.batchsize or 1. There might be a more legible way of getting this.
  
  local targindices = {}
  for _,word in ipairs(opt.targwords) do
    if opt.vocab[word] then
      targindices[opt.vocab[word]] = true
    else
      error("Word '" .. word .. "' not in vocabulary!")
    end
  end
  
  -- Input Looks Like: 
  -- --------------Forward Inputs -------------------------------->
  -- <null>, <null>, <s>,  fill,     my,   box, with, five, dozen, liquor, jugs
  -- <null>, <null>, </s>, jugs, liquor, dozen, five, with,   box,     my, fill
  -- --------------Backward Inputs ------------------------------->
  --   -               + bwd[seqlen]       -> fwd[pad+1]
  -- fwd[pad+1]        + bwd[seqlen-1]     -> fwd[pad+2]      == bwd[seqlen]
  -- fwd[pad+2]        + bwd[seqlen-2]     -> fwd[pad+3]      == bwd[seqlen-1]
  -- fwd[seqlen-1]     + bwd[pad+1]        -> fwd[seqlen]     == bwd[pad+2]
  -- fwd[seqlen]       +    -              -> bwd[pad+1]
  -- fwd[pad+n]        + bwd[seqlen-n]     -> fwd[pad+n+1]    == bwd[seqlen+1-n]
  -- Where seqlen is the tensor size in the time dimension and pad is the number of elements filled with 0 padding

  -- Target Looks Like:
  --       | ----Sequence Length----------------- |
  --             Forward Targets
  -- --------------------------------------------------------------->
  --     1      2    3    4    5     6     7      8       9       10          Index
  -- <null>, <null>, my, box, with, five, dozen, liquor, jugs,   </s>         Token
  --    <s>,   fill, my, box, with, five, dozen, liquor, <null>, <null>       Token
  --     10     9    8    7    6     5     4      3       2        1          Index
  --    <--------------------------------------------------------------
  --            Backward Targets
  -- | ---------Sequence Length-------------- |
  -- fwd[n] = bwd[seqlen-n], 1 <= n < seqlen
  -- Seqlen = First null index -1

  local proc_sm_inputs = torch.Tensor(seqlen, batchsize, opt.sm_inputsize):zero()
  local proc_sm_targets = torch.Tensor(seqlen, batchsize):zero()
  -- 2 is the class for "no".
  -- 1 is the class for "yes".
  -- 0 is a non-class used for masking things to prevent learning where a timestep is empty for a given sequence.

  --Format: sm_input[direction][layer][timestep][batch][neuron]
  --                      Just Some Tables Here |We Tensor Now

  -- Get all targets from the forward reader targets.
  -- Because of the extra <s> and </s> tokens, it contains all the tokens in the actual sequence (plus </s>)
  for bseq=1,batchsize do --batch sequence
    local padding = pads[bseq]
    for step=1+padding,seqlen do
      if targindices[fw_trget[step][bseq]] ~= nil then -- evaluates to true if word with index fw_trget[step][bseq] is in the list of target words
        proc_sm_targets[step][bseq] = 1
      else
        proc_sm_targets[step][bseq] = 2
      end
    end
  end

  local startindex = 1

  -- cantrip starts here
  for layer=1,#sm_input[1] do
    local fw_end_index = startindex-1 + sm_input[1][layer][1]:size(2) -- Define the space for forward-reading neurons this timestep. -1 because Lua's indices start at 1.
    local bw_end_index = fw_end_index + sm_input[2][layer][1]:size(2)
    for bseq=1,batchsize do
      local padding = pads[bseq]
      for step=1+padding,seqlen do
        proc_sm_inputs[{step,bseq,{startindex,     fw_end_index}}] = (sm_input[1][layer][step]                 [bseq]):type('torch.FloatTensor')
        proc_sm_inputs[{step,bseq,{fw_end_index+1, bw_end_index}}] = (sm_input[2][layer][seqlen-(step-padding)+1][bseq]):type('torch.FloatTensor')
      end
    end
    startindex = bw_end_index + 1
  end
-- end cantrip

  return proc_sm_inputs, proc_sm_targets
end

--[[
END Utility functions
--]]

--[[
BEGIN RNN functions
--]]

function lm_build(filename)
  -- Build a language model. If provided, filename determines which file to load (forward or backward, in this case).
  -- Returns the model, a target module for formatting targets returned by the batchloader, a criterion for computing losses,
  -- and an optimState table for defining optimizer behavior.
  local lang_mod, target_mod, criterion, optimState

  if opt.loadlm ~= '' and filename then
    lang_mod = torch.load(opt.loadlm .. filename .. '.t7')
    if opt.cuda then
      lang_mod = lang_mod:cuda()
    end
  else
    lang_mod = nn.Sequential()

    if opt.cuda then
      lang_mod:add(nn.Convert())
    end

    local inputsize = opt.lm_inputsize
    for i,hiddensize in ipairs(opt.lm_hiddensize) do
      local rnn
      rnn = nn.SeqLSTM(inputsize, hiddensize) -- TODO: Consider SeqLSTMP
      rnn.maskzero = true
      lang_mod:add(rnn)

      if opt.lm_dropout > 0 and opt.lm_dropout < 1 then
        lang_mod:add(nn.Dropout(opt.lm_dropout))
      end

      inputsize = hiddensize
    end
    lang_mod:add(nn.SplitTable(1,3)) -- Split output along timesteps into table

    lang_mod:add(nn.Sequencer(nn.MaskZero(nn.Linear(inputsize, #opt.vocab), 1)))
    lang_mod:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1)))

    lang_mod:remember('neither')

    if opt.uniform > 0 then
      for k,param in ipairs(lang_mod:parameters()) do
        param:uniform(-opt.uniform, opt.uniform)
      end
    end
    lang_mod.parameters, lang_mod.gradParameters = lang_mod:getParameters()
  end

  if not opt.silent then
    print("Language Model: ")
    print(lang_mod)
  end

  target_mod = nn.Sequential()
  target_mod:add(nn.SplitTable(1,3)) -- Split tensor along dimension 1 (timesteps), accepting up to 3D tensors

  local crit = nn.ClassNLLCriterion() -- Passing frequencies here makes things worse.
  crit.sizeAverage = false
  criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(crit,1))

  optimState = {  learningRate = opt.lm_lr,
    beta1 = opt.lm_momentum,
    weightDecay = opt.lm_weightdecay
  }

  if opt.cuda then
    lang_mod:cuda()
    target_mod:cuda()
    criterion:cuda()
  end  

  return lang_mod, target_mod, criterion, optimState
end

function sm_build(filename) --TODO: Remove filename argument; it can be hardcoded, and was only necessary to distinguish between forward and backward in lm_build.
  local sal_mod, target_mod, criterion, optimState

  if opt.loadsm ~= '' and filename then
    sal_mod = torch.load(opt.loadsm .. filename .. '.t7')
    if opt.cuda then
      sal_mod = sal_mod:cuda()
    end
  else
    if not filename then
      print("No filename supplied! Initializing a new model.")
    end
    sal_mod = nn.Sequential()

    if opt.cuda then
      sal_mod:add(nn.Convert())
    end

    local inputsize = opt.sm_inputsize

    for i,hiddensize in ipairs(opt.sm_hiddensize) do
      local layer = nn.Sequential()
      layer:add(nn.MaskZero(nn.Linear(inputsize, hiddensize), 1))
      layer:add(nn.MaskZero(nn.ReLU(true), 1)) -- true to do operation in place instead of making a new tensor. Memory savings!
      if opt.sm_dropout > 0 and opt.sm_dropout < 1 then
        layer:add(nn.Dropout(opt.sm_dropout))
      end
      sal_mod:add(nn.Sequencer(layer))
      inputsize = hiddensize
    end
    sal_mod:add(nn.Sequencer(nn.MaskZero(nn.Linear(inputsize, 2), 1)))
    sal_mod:add(nn.Sequencer(nn.MaskZero(nn.LogSoftMax(), 1)))
    sal_mod:add(nn.SplitTable(1,3))

    if opt.uniform > 0 then
      for k,param in ipairs(sal_mod:parameters()) do
        param:uniform(-opt.uniform, opt.uniform)
      end
    end

    sal_mod.parameters, sal_mod.gradParameters = sal_mod:getParameters()
  end

  if not opt.silent then
    print("Salience Model: ")
    print(sal_mod)
  end

  target_mod = nn.Sequential()
  target_mod:add(nn.SplitTable(1,3))

  local crit = nn.ClassNLLCriterion()
  crit.sizeAverage = false
  criterion = nn.SequencerCriterion(nn.MaskZeroCriterion(crit,1))

  optimState = {  learningRate = opt.sm_lr,
    beta1 = opt.sm_momentum,
    weightDecay = opt.sm_weightdecay
  }
  
  if opt.cuda then
    sal_mod:cuda()
    target_mod:cuda()
    criterion:cuda()
  end

  return sal_mod, target_mod, criterion, optimState
end

function lm_train(fwd, bwd)
  --Takes a table containing a forward-reading language model and related objects, and a backward-reading version of same
  --Executes a single training epoch
  unset_record(fwd.model)
  unset_record(bwd.model)
  local timer = torch.Timer()
  local batchcounter = 1
  local fw_trainLoss = 0 -- loss on this epoch
  local bw_trainLoss = 0
  local fw_pred_counter = 0
  local bw_pred_counter = 0
  local fw_fail = 0
  local bw_fail = 0
  local trainprogress = -1
  local batches
  fwd.model:training()   
  bwd.model:training()
  while trainprogress ~= 0 do
    if trainprogress < 0 then trainprogress = 0 end -- Starts with trainprogress = -1, but that's an invalid starting cursor.
    batches, trainprogress = batches_load('train_data.txt', opt.batchsize, opt.batchnum, trainprogress)
    batches = batches_perm(batches)
    for i=1,#batches do
      local fw_input, bw_input, fw_trget, bw_trget = batches[i].fwd_in, batches[i].bwd_in, batches[i].fwd_targ, batches[i].bwd_targ

      -- Batch Forward-reader Input
      -- Batch Backward-reader Input
      -- Batch Forward-reader Target
      -- Batch Backward-reader Target
      
      local fw_feval = function()
        fwd.model.gradParameters:zero()
        fw_trget = fwd.targmod:forward(fw_trget)
        local outputs = fwd.model:forward(fw_input)

        for t=1,#outputs do
          local _,pred = torch.max(outputs[t], 2)
          for b=1,opt.batchsize do
            if fw_trget[t][b] ~= 0 then
              fw_pred_counter = fw_pred_counter + 1
              if(pred[b][1] ~= fw_trget[t][b]) then
                fw_fail = fw_fail + 1
              end
            end  
          end
        end
        
        local err = fwd.crit:forward(outputs, fw_trget)
        fw_trainLoss = fw_trainLoss + err
        local gradOutputs = fwd.crit:backward(outputs, fw_trget)

        fwd.model:backward(fw_input, gradOutputs)
        return err, fwd.model.gradParameters
      end
      local bw_feval = function() 
        -- I am 90% sure that there's a way to make a closure that handles both models at once
        -- This is definitely a TODO:
        bwd.model.gradParameters:zero()
        bw_trget = bwd.targmod:forward(bw_trget)
        local outputs = bwd.model:forward(bw_input)

        for b=1,opt.batchsize do
          for t=1,#outputs do
            if bw_trget[t][b] ~= 0 then
              bw_pred_counter = bw_pred_counter + 1
              local _,pred = torch.max(outputs[t], 2)
              if(pred[b][1] ~= bw_trget[t][b]) then
                bw_fail = bw_fail + 1
              end
            end            
          end
        end

        local err = bwd.crit:forward(outputs, bw_trget)

        bw_trainLoss = bw_trainLoss + err
        local gradOutputs = bwd.crit:backward(outputs, bw_trget)
        bwd.model:backward(bw_input, gradOutputs)
        return err, bwd.model.gradParameters
      end
      optim.adam(fw_feval, fwd.model.parameters, fwd.optim) -- Do optimization on forward reader
      optim.adam(bw_feval, bwd.model.parameters, bwd.optim) -- Do optimization on backward reader
      if batchcounter % 1000 == 0 then
        collectgarbage()
      end
      
      if not opt.silent and opt.t_printfreq > 0 and (batchcounter) % opt.t_printfreq == 0 then
        --TODO: format strings instead of clumsy whatever-this-is
        print("Batch ", batchcounter, "/", opt.trainsize, "FW % Failure ", 100 * fw_fail/fw_pred_counter, 
          "BW % Failure ", 100 * bw_fail/bw_pred_counter, "Sequence Length", fw_input:size(1)) 
      end
      
      batchcounter = batchcounter + 1
    end
  end
  if cutorch ~= nil then cutorch.synchronize() end
  local speed = timer:time().real/batchcounter
  if not opt.silent then print(string.format("Speed: %f sec/batch ", speed)) end
  return fw_trainLoss, bw_trainLoss
end

function sm_train(sm, fwd, bwd)
  fwd.model:evaluate()
  set_record(fwd.model)
  bwd.model:evaluate()
  set_record(bwd.model)
  sm.model:training()
  local timer = torch.Timer()
  local batchcounter = 1
  local trainLoss = 0
  local trainprogress = -1
  local batches
  while trainprogress ~= 0 do
    if trainprogress < 0 then trainprogress = 0 end -- Correct the initial value (see lm_train())
    batches, trainprogress = batches_load('train_data.txt', opt.batchsize, opt.batchnum, trainprogress)
    batches = batches_perm(batches)
    for i=1,#batches do
      local fw_input, bw_input, fw_trget, bw_trget, pads = batches[i].fwd_in, batches[i].bwd_in, batches[i].fwd_targ, batches[i].bwd_targ, batches[i].pads
      
      active_history = {{}}            -- Add the first table so the forward reader can store its states
      fwd.model:forward(fw_input)
      table.insert(active_history, {}) -- Add the second table for the backward reader to fill up
      bwd.model:forward(bw_input)
      local sm_input, sm_trget = sm_process_input(active_history, fw_trget, pads)

      local sm_feval = function()
        sm.model.gradParameters:zero()
        local outputs = sm.model:forward(sm_input)

        for t=1,#outputs do
          local _,pred = torch.max(outputs[t], 2) 
          -- Gets a batchsize x 1 tensor of the highest-probability classes at this timestep
          for b=1,opt.batchsize do
            if fw_trget[t][b] ~= 0 and pred[b][1] == 1 then
              local predstr = opt.ivocab[fw_trget[t][b]]
              opt.sal_positives[predstr] = opt.sal_positives[predstr] and opt.sal_positives[predstr] + 1 or 1
              --If exists, increment. Else initialize.
            end
          end
        end

        sm_trget = sm.targmod:forward(sm_trget)
        local err = sm.crit:forward(outputs, sm_trget)

        trainLoss = trainLoss + err
        local gradOutputs = sm.crit:backward(outputs, sm_trget)
        sm.model:backward(sm_input, gradOutputs)
        return err, sm.model.gradParameters
      end 
      optim.adam(sm_feval, sm.model.parameters, sm.optim)
      if batchcounter % 1000 == 0 then
        collectgarbage()
      end
      
      if not opt.silent and opt.t_printfreq > 0 and (batchcounter) % opt.t_printfreq == 0 then
        print("Batch ", batchcounter, "/", opt.trainsize, "Avg Loss ", trainLoss/batchcounter, "Sequence Length", fw_input:size(1)) 
      end
      
      batchcounter = batchcounter + 1
    end
  end

  if not opt.silent then 
    for i,j in pairs(opt.sal_positives) do
      print(i, j)
    end
  end
  
  if cutorch then cutorch.synchronize() end
  local speed = timer:time().real/batchcounter
  if not opt.silent then print(string.format("Speed: %f sec/batch ", speed)) end
  return trainLoss
end

function lm_valtst(fwd, bwd, datafile)
-- Takes the same arguments as lm_train(), plus the name of the file to load
-- datafile is either your validation or your test set

  local timer = torch.Timer()
  local batchcounter = 1 -- TODO: rename this to seqcounter
  local fw_loss = 0
  local bw_loss = 0
  local fw_pred_counter = 0
  local bw_pred_counter = 0
  local fw_fail = 0
  local bw_fail = 0
  local trainprogress = -1
  local batches -- A bit redundant; each batch is only 1 sequence
  -- TODO: Replace batchsize with 1 or eliminate it, as appropriate
  fwd.model:evaluate()
  unset_record(fwd.model)
  bwd.model:evaluate()
  unset_record(bwd.model)
  while trainprogress ~= 0 do
    if trainprogress < 0 then trainprogress = 0 end -- Correct the initial value (see lm_train())
    batches, trainprogress = batches_load(datafile, 1, opt.batchnum, trainprogress) -- batch size 1
    for i=1,#batches do
      local fw_input, bw_input, fw_trget, bw_trget = batches[i].fwd_in, batches[i].bwd_in, batches[i].fwd_targ, batches[i].bwd_targ

      fw_trget = fwd.targmod:forward(fw_trget)
      local outputs = fwd.model:forward(fw_input)

      for b=1,1 do -- There's only 1 sequence in these "batches".
        for t=1,#outputs do
          if fw_trget[t][b] ~= 0 then
            fw_pred_counter = fw_pred_counter + 1
            local _,pred = torch.max(outputs[t], 2)
            if(pred[b][1] ~= fw_trget[t][b]) then
              fw_fail = fw_fail + 1
            end
          end            
        end
      end

      local err = fwd.crit:forward(outputs, fw_trget)
      fw_loss = fw_loss + err

      bw_trget = bwd.targmod:forward(bw_trget)
      outputs = bwd.model:forward(bw_input) -- Overwrite the fwd pass. Not relevant anymore.

      for b=1,1 do
        for t=1,#outputs do
          if bw_trget[t][b] ~= 0 then
            bw_pred_counter = bw_pred_counter + 1
            local _,pred = torch.max(outputs[t], 2)
            if(pred[b][1] ~= bw_trget[t][b]) then
              bw_fail = bw_fail + 1
            end
          end            
        end
      end      

      err = bwd.crit:forward(outputs, bw_trget)
      bw_loss = bw_loss + err

      if batchcounter % 1000 == 0 then
        collectgarbage()
      end
      
      local size = datafile == 'val_data.txt' and opt.validsize or opt.testsize
      if not opt.silent and opt.v_printfreq > 1 and (batchcounter) % opt.v_printfreq == 0 then
        print("Sequence ", batchcounter, "/", size, "FW % Failure ", 100* fw_fail/fw_pred_counter, 
          "BW % Failure ", 100 * bw_fail/bw_pred_counter, "Sequence Length", fw_input:size(1)) 
      end
      
      batchcounter = batchcounter + 1
      
    end
  end
  if cutorch then cutorch.synchronize() end
  local speed = timer:time().real/batchcounter
  if not opt.silent then print(string.format("Speed: %f sec/sequence ", speed)) end
  return fw_loss, bw_loss
end


function sm_valtst(sm, fwd, bwd, datafile)
  -- Does a validation of the salience model
  local timer = torch.Timer()
  local batchcounter = 1 -- TODO: Rename this seqcounter
  local loss = 0 -- loss on this epoch
  local trainprogress = -1
  local batches
  -- TODO: Replace batchsize with 1 or eliminate it, as appropriate
  fwd.model:evaluate()
  set_record(fwd.model)
  bwd.model:evaluate()
  set_record(bwd.model)
  sm.model:evaluate()   
  while trainprogress ~= 0 do
    if trainprogress < 0 then trainprogress = 0 end -- Correct initial value
    batches, trainprogress = batches_load(datafile, 1, opt.batchnum, trainprogress) -- batch size 1

    for i=1,#batches do
      local fw_input, bw_input, fw_trget, bw_trget, pads = batches[i].fwd_in, batches[i].bwd_in, batches[i].fwd_targ, batches[i].bwd_targ, batches[i].pads

      active_history = {{}}
      fwd.model:forward(fw_input)
      table.insert(active_history, {})
      bwd.model:forward(bw_input)
      local sm_input, sm_trget = sm_process_input(active_history, fw_trget, pads)

      local outputs = sm.model:forward(sm_input)

      for t=1,#outputs do
        local _,pred = torch.max(outputs[t], 2)
        for b=1,1 do
          if fw_trget[t][b] ~= 0 and pred[b][1] == 1 then
            local predstr = opt.ivocab[fw_trget[t][b]]
            opt.sal_positives[predstr] = opt.sal_positives[predstr] and opt.sal_positives[predstr] + 1 or 1 --If exists, increment. Else initialize.
          end
        end
      end

      sm_trget = sm.targmod:forward(sm_trget)
      local err = sm.crit:forward(outputs, sm_trget)
      loss = loss + err

      if batchcounter % 1000 == 0 then
        collectgarbage()
      end
      local size = datafile == 'val_data.txt' and opt.validsize or opt.testsize
      if not opt.silent and opt.v_printfreq > 0 and (batchcounter) % opt.v_printfreq == 0 then
        print("Sequence ", batchcounter, "/", size, "Avg Loss ", loss/batchcounter, 
          "Sequence Length", fw_input:size(1)) 
      end
      batchcounter = batchcounter + 1
      
    end
  end

  if not opt.silent then
    for i,j in pairs(opt.sal_positives) do
      print(i, j)
    end
  end

  if cutorch then cutorch.synchronize() end
  local speed = timer:time().real/batchcounter
  if not opt.silent then print(string.format("Speed: %f sec/sequence ", speed)) end
  return loss
end

function lm_sample(lm)
  lm:evaluate()
  local beginning = 0
  local input = torch.Tensor(1, 1, opt.lm_inputsize):zero()
  local output, pred_word
  local outstr = ""
  for _,word in ipairs(opt.seedwords) do
    input[1][1] = opt.embeds[word]
    outstr = outstr .. word .. " "
    output = lm:forward(input)
  end
  if not opt.silent then
    print("Sampling Language Model With Seed: " .. outstr)
  end
  while beginning < opt.samplemax and pred_word ~= '<s>' and pred_word ~= '</s>' do
    beginning = beginning + 1
    local _,pred_index = torch.max(output[1], 2) -- output is a table with 1 element; a tensor with the scores predicted by the network for each vocabulary element. The first return value is the value of the highest score; discard it. The second argument causes max to return, as its second return value, a tensor whose only element is the corresponding index; this is the useful value.
    pred_word = opt.ivocab[pred_index[1][1]]
    outstr = outstr .. pred_word .. " "
    input[1][1] = opt.embeds[pred_word] -- Feed result back into the net
    output = lm:forward(input)
  end
  if not opt.silent then
    print(outstr)
  end
end

main()