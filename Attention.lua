require 'nn';
require 'nngraph';
require 'AddBias';
require 'RNNAttention2';
require 'Recurrent2';
require 'MonotonicAlignment2';
require 'Squeeze';
require 'ExpandAs';
require 'AddDim';
require 'TemporalConvolutionZeroBias';

--nngraph.setDebug(true)
local Attention, parent = torch.class('nn.Attention','nn.Module')

function Attention:__init(decoder_recurrent,		-- recurrent part of the decoder ~ f
						  decoder_mlp,				-- MLP that converts state to y ~ g
						  scoreDepth,				-- length of input vector to e
						  hybridAttendFilterSize,	-- size of conv kernel on prev alpha
						  hybridAttendFeatureMaps,  -- # kernels in conv on prev alpha
						  stateDepth,				-- size of hidden state
						  annotationDepth,			-- size of annotations
						  outputDepth,				-- size of output layer
						  monoAlignPenalty,         -- monotonic alignment penalty
						  penaltyLambda)            -- strength of penalty
						  --T							-- length of output sequence
						  --)
	parent.__init(self)

	self.decoder_recurrent = decoder_recurrent
	self.scoreDepth = scoreDepth
	self.hybridAttendFilterSize = hybridAttendFilterSize
	self.hybridAttendFeatureMaps = hybridAttendFeatureMaps
	self.stateDepth = stateDepth
	self.annotationDepth = annotationDepth
	self.MonotonicAlignmentPenalty = monoAlignPenalty or false
	self.lambda = penaltyLambda or 0.0
	--self.T = T

	------------------ construct attentional decoder ------------------
	-- First, construct Vh separately, which will be less memory intensive than
	-- if we add it to the RNN
	------------------ Vh ------------------
	local h 			= nn.Identity()()
	local _Vh = nn.TemporalConvolutionZeroBias(annotationDepth,scoreDepth,1)(h)
	local Vh = nn.gModule({h},{_Vh})
	Vh.name = "Vh"
	self.Vh = Vh

	-- Next, construct the decoder
	------------------ inputs ------------------
	local input             = nn.Identity()()
	local prev_hidden 		= nn.Identity()()

	local nonrecurrent, prev_y       = input:split(2)
	local prev_alpha,prev_s,prev_mem = prev_hidden:split(3)
	-- prev_alpha ~ L 
	-- prev_s     ~ stateDepth 
	-- prev_mem   ~ stateDepth
	local Vh_inp, h = nonrecurrent:split(2)

	------------------ Ws ------------------
	local prev_s_reshaped = nn.View(stateDepth,1)(prev_s)
	local ws = nn.TemporalConvolution(1,scoreDepth,stateDepth)(prev_s_reshaped)
	local Ws = nn.ExpandAs(1,2)({ws,Vh_inp})
	--local Ws = nn.Reshape(L,scoreDepth)(nn.Replicate(L,1,2)(ws))
	self.ws = ws
	Ws.name = 'Ws'
	--  L x scoreDepth
	
	------------------ UF ------------------
	local UF, Z
	if hybridAttendFeatureMaps > 0 then
		local pad_left,pad_right
		if hybridAttendFilterSize % 2 == 1 then
			-- odd
			pad_left = math.floor((hybridAttendFilterSize-1)/2)
			pad_right = pad_left
		else
			-- even
			pad_left = hybridAttendFilterSize/2
			pad_right = pad_left-1
		end
		local prev_alpha_reshaped = nn.AddDim(1,1)(prev_alpha)
		-- alphaReshaped ~ L x 1
		local padded_alpha = nn.Padding(1,pad_right,2)(nn.Padding(1,-pad_left,2)(prev_alpha_reshaped))
		padded_alpha.name = "padded alpha"
		local F = nn.TemporalConvolution(1,hybridAttendFeatureMaps,hybridAttendFilterSize)(padded_alpha)
		UF = nn.TemporalConvolutionZeroBias(hybridAttendFeatureMaps,scoreDepth,1)(F)
		UF.name = 'UF'
		self.UF = UF
		--  L x scoreDepth
	    Z = nn.CAddTable()({Ws,Vh_inp,UF})
		--  L x scoreDepth
	else
	    Z = nn.CAddTable()({Ws,Vh_inp})
	end
	self.Z = Z
	Z.name = 'Z'
	
	------------------ tanh ------------------
	local tanh = nn.Tanh()(Z)
	tanh.name = 'tanh'
	self.tanh = tanh
	--  L x scoreDepth

	------------------ e_t ------------------
	local e = nn.TemporalConvolutionZeroBias(scoreDepth,1,1)(tanh)
	e.name = 'e'
	self.e = e
	--  L x 1

	------------------ alpha_t ------------------
	--local alpha = nn.SoftMax()(nn.View(L)(e))
	local alpha = nn.SoftMax()(nn.Squeeze(2,2)(e))
	alpha.name = 'alpha'
	--  L

	------------------ monotonic alignment penalty ------------------
	if self.MonotonicAlignmentPenalty then
		print('using monotonic alignment penalty ' .. self.lambda)
		alpha = nn.MonotonicAlignment(self.lambda)({alpha,prev_alpha})
	end
	alpha.name = 'penalty'
	--  L

	------------------ c_t ------------------
	--  alpha ~ L
	--  h     ~ L x annotationDepth
	local alpha_view = nn.Replicate(1,1,1)(alpha)
	--  1 x L
	local c = nn.View(-1):setNumInputDims(2)(nn.MM()({alpha_view,h}))
	self.c = c
	--  annotationDepth

	------------------ decoder_recurrent ------------------
	-- inputs:
	-- 	 c_t	   ~ input       	 ~ annotationDepth
	-- 	 y_{t-1}   ~ input      	 ~ outputDepth
	-- 	 s_{t-1}   ~ prev_output     ~ stateDepth
	-- 	 mem_{t-1} ~ prev_memory     ~ stateDepth
	-- outputs:
	--   s_t	   ~ output			 ~ stateDepth
	--   mem_t     ~ memory			 ~ stateDepth
	--local y_in        = nn.Tanh()(nn.Linear(outputDepth,stateDepth)(prev_y))
	--local c_in        = nn.Tanh()(nn.Linear(annotationDepth,stateDepth)(c))
	local y_in        = nn.Linear(outputDepth,stateDepth)(prev_y)
	local c_in        = nn.Linear(annotationDepth,stateDepth)(c)
	local dec_rec_inp = nn.Linear(2*stateDepth,stateDepth)(nn.JoinTable(1,1)({c_in,y_in}))
	local s,mem       = decoder_recurrent({dec_rec_inp,prev_s,prev_mem}):split(2)
	decoder_recurrent.name = 'decoder_recurrent'

	------------------ decoder_mlp ------------------
	-- inputs:
	--   s_t       ~ input      ~ stateDepth
	-- 	 c_t	   ~ input      ~ annotationDepth
	-- outputs:
	--   y_t	   ~ output	    ~ outputDepth
	local y    		= decoder_mlp({s,c})

	------------------ decoder_base ------------------
	-- decoder_base = attention + recurrent + mlp
	--
	-- inputs:
	--   nonrecurrent ~ input	 ~ {Vh(h),h}
	--   y_{t-1}      ~ output   ~ outputDepth
	--   alpha_{t-1}  ~ hidden	 ~ L (encoder length)
	--   s_{t-1}      ~ hidden   ~ stateDepth
	--   mem_{t-1}    ~ hidden   ~ stateDepth
	-- outputs:
	--   alpha_t 	  ~ hidden	 ~ L (encoder length)
	--   s_t     	  ~ hidden   ~ stateDepth
	--   mem_t   	  ~ hidden   ~ stateDepth
	--   y_t     	  ~ output   ~ outputDepth
	local hidden		= nn.Identity()({alpha, s, mem})
	local decoder_base_	= nn.gModule({input,prev_hidden},{y,hidden})
	decoder_base_.name = "decoder_base_"
	self.decoder_base_ = decoder_base_

	local dimhidden 	= {0,stateDepth,stateDepth}
	local dimoutput 	= outputDepth 
	local decoder_base  = nn.Recurrent(decoder_base_,dimhidden,dimoutput)
	self.decoder_base = decoder_base

	------------------ encoder output + decoder base ------------------
	-- inputs:
	--   h		     ~ input	 ~ annotationDepth
	--   Vh(h)	     ~ input	 ~ scoreDepth
	-- outputs:			
	--   y_{1:T}      ~ output    ~ T x outputDepth
	local numRecurrentInputs = 0
	local numNonRecurrentInputs = 2

	local h		  = nn.Identity()()
	local y       = nn.Identity()()
	local rnn_inp = {nn.Identity()({Vh(h),h}),y}
	local rnn	  = nn.RNNAttention(decoder_base,outputDepth,false)
	self.rnn = rnn
	--nngraph.annotateNodes()
	local decoder = nn.gModule({h,y},{rnn(rnn_inp)}) 
	decoder.name = "decoder"

	self.decoder = decoder
end

function Attention:getRNNlayer(layername)
	local rnn = self.rnn
	local layer = {}
	local sequence_dim = rnn.sequence_dim
	local batchSize = rnn.batchSize
	assert(batchSize ~= nil, 'forward must be run at least once to recover ' .. layername)
	
	for t = 1, rnn.T do
		local nodes = rnn.rnn[t].recurrent.backwardnodes
		for i = 1, #nodes do
			f = nodes[i]
			if f.name == layername then
				local output = f.data.module.output
				local size   = output:size():totable()
				if batchSize == 0 then
					layer[t] = output:view(1,unpack(size))
				else
					local b = table.remove(size,1)
					assert(b == batchSize, 'inconsistent tensor sizes')
					layer[t] = output:view(batchSize,1,unpack(size))
				end
			end
		end
	end
	return nn.JoinTable(sequence_dim):type(layer[1]:type()):forward(layer)
end

function Attention:alpha()
	return self:getRNNlayer('alpha')
end

function Attention:penalty()
	return self:getRNNlayer('penalty')
end
function Attention:Ws()
	return self:getRNNlayer('Ws')
end

function Attention:setpenalty(penalty)
	local rnn = self.rnn
    local nodes = rnn.recurrent.recurrent.backwardnodes
	local foundpenalty = false
    for i = 1, #nodes do
        local f = nodes[i]
        if f.name == 'penalty' then
            f.data.module.lambda = opt.penalty
            print('setting penalty to ' .. opt.penalty)
			foundpenalty = true
        end
    end
	assert(foundpenalty == true, 'could not find penalty node')
    for t = 1, rnn.T do
        local nodes = rnn.rnn[t].recurrent.backwardnodes
        for i =1, #nodes do
            local f = nodes[i]
            if f.name == 'penalty' then
                f.data.module.lambda = opt.penalty
            end
        end
    end
end

function Attention:parameters()
	return self.decoder:parameters()
end

function Attention:training()
	self.decoder:training()
end

function Attention:evaluate()
	self.decoder:evaluate()
end

function Attention:double()
	self.decoder = self.decoder:double()
	return self:type('torch.DoubleTensor')
end
function Attention:float()
	self.decoder = self.decoder:float()
	return self:type('torch.FloatTensor')
end
function Attention:cuda()
	self.decoder = self.decoder:cuda()
	return self:type('torch.CudaTensor')
end

function Attention:setT(T)
	self.rnn:setT(T)
end

function Attention:updateOutput(input)
	local x,y = unpack(input)
	local L,T
	if x:nDimension() == 2 then
		-- nonbatch mode
		L = x:size(1)
		T = y:size(1)
	elseif x:nDimension() == 3 then
		L = x:size(2)
		T = y:size(2)
	else
		error('x must be 2d or 3d')
	end
	self.rnn:apply2clones(function(x) x.dimhidden = {L,self.stateDepth,self.stateDepth} end)
	self:setT(T)
	self.output = self.decoder:forward(input)
	return self.output
end

function Attention:updateGradInput(input, gradOutput)
	self.gradInput = self.decoder:backward(input, gradOutput)
	return self.gradInput
end
		





