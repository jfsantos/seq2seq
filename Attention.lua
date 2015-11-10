require 'nn';
require 'nngraph';
require 'AddBias';
require 'RNN3';
require 'Recurrent';

nngraph.setDebug(true)
local Attention, parent = torch.class('nn.Attention','nn.Module')

function Attention:__init(decoder_recurrent,		-- recurrent part of the decoder ~ f
						  decoder_mlp,				-- MLP that converts state to y ~ g
						  scoreDepth,				-- length of input vector to e
						  hybridAttendFilterSize,	-- size of conv kernel on prev alpha
						  hybridAttendFeatureMaps,  -- # kernels in conv on prev alpha
						  stateDepth,				-- size of hidden state
						  annotationDepth,			-- size of annotations
						  outputDepth,				-- size of output layer
						  L,						-- length of annotations
						  T							-- length of output sequence
						  )
	parent.__init(self)

	self.decoder_recurrent = decoder_recurrent
	self.scoreDepth = scoreDepth
	self.hybridAttendFilterSize = hybridAttendFilterSize
	self.hybridAttendFeatureMaps = hybridAttendFeatureMaps
	self.stateDepth = stateDepth
	self.annotationDepth = annotationDepth
	self.L = L
	self.T = T


	------------------ construct attentional decoder ------------------
	-- First, construct Vh separately, which will be less memory intensive than
	-- if we add it to the RNN
	------------------ Vh ------------------
	local h 			= nn.Identity()()
	local _Vh = nn.TemporalConvolution(annotationDepth,scoreDepth,1)(h)
	local Vh = nn.gModule({h},{_Vh})
	Vh.name = "Vh"
	self.Vh = Vh

	-- Next, construct the decoder
	------------------ inputs ------------------
	local nonrecurrent		= nn.Identity()()
	local prev_y			= nn.Identity()()
	local prev_hidden 		= nn.Identity()()

	local prev_alpha,prev_s,prev_mem = prev_hidden:split(3)
	-- prev_alpha ~ L 
	-- prev_s     ~ stateDepth 
	-- prev_mem   ~ stateDepth
	local Vh_inp, h = nonrecurrent:split(2)
	
	------------------ UF ------------------
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
	local prev_alpha_reshaped = nn.Reshape(L,1)(prev_alpha)
	-- alphaReshaped ~ L x 1
	local padded_alpha = nn.Padding(1,pad_right,2)(nn.Padding(1,-pad_left,2)(prev_alpha_reshaped))
	padded_alpha.name = "padded alpha"
	local F = nn.TemporalConvolution(1,hybridAttendFeatureMaps,hybridAttendFilterSize)(padded_alpha)
	local UF = nn.TemporalConvolution(hybridAttendFeatureMaps,scoreDepth,1)(F)
	--  L x scoreDepth
	
	------------------ Ws ------------------
	local prev_s_reshaped = nn.Reshape(stateDepth,1)(prev_s)
	local ws = nn.TemporalConvolution(1,scoreDepth,stateDepth)(prev_s_reshaped)
	local Ws = nn.Reshape(L,scoreDepth)(nn.Replicate(L,1,2)(ws))
	--  L x scoreDepth
	
	------------------ tanh ------------------
	local Z = nn.CAddTable()({Ws,Vh_inp,UF})
	local tanh = nn.Tanh()(nn.AddBias(scoreDepth,false)(Z))
	--  L x scoreDepth

	------------------ e_t ------------------
	local e = nn.TemporalConvolution(scoreDepth,1,1)(tanh)
	--  L x 1

	------------------ alpha_t ------------------
	local alpha = nn.SoftMax()(nn.Reshape(L)(e))
	--  L

	------------------ c_t ------------------
	local c = nn.View(-1):setNumInputDims(2)(nn.MM()({nn.Reshape(1,L)(alpha),h}))
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
	local dec_rec_inp = {nn.Identity()({c,prev_y}),prev_s,prev_mem}
	local s,mem 	  = decoder_recurrent(dec_rec_inp):split(2)

	------------------ decoder_recurrent ------------------
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
	--   alpha_{t-1}  ~ hidden	 ~ L (encoder length)
	--   s_{t-1}      ~ hidden   ~ stateDepth
	--   mem_{t-1}    ~ hidden   ~ stateDepth
	--   y_{t-1}      ~ output   ~ outputDepth
	-- outputs:
	--   alpha_t 	  ~ hidden	 ~ L (encoder length)
	--   s_t     	  ~ hidden   ~ stateDepth
	--   mem_t   	  ~ hidden   ~ stateDepth
	--   y_t     	  ~ output   ~ outputDepth
	local hidden		= nn.Identity()({alpha, s, mem})
	local decoder_base_	= nn.gModule({nonrecurrent,prev_y,prev_hidden},{y,hidden})
	decoder_base_.name = "decoder_base_"
	self.decoder_base_ = decoder_base_

	local dimhidden 	= {L,stateDepth,stateDepth}
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
	local rnn_inp = {Vh(h),h}
	local rnn	  = nn.RNN3(decoder_base,T,outputDepth,false)(rnn_inp)
	self.rnn = rnn
	nngraph.annotateNodes()
	local decoder = nn.gModule({h},{rnn}) 
	decoder.name = "decoder"

	self.decoder = decoder
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


function Attention:updateOutput(input)
	self.output = self.decoder:forward(input)
	return self.output
end

function Attention:updateGradInput(input, gradOutput)
	self.gradInput = self.decoder:backward(input, gradOutput)
	return self.gradInput
end
		





