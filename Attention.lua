require 'nn';
require 'nngraph';
require 'AddBias';

local Attention, parent = torch.class('nn.Attention','nn.Module')

function Attention:__init(scoreDepth,
						  hybridAttendFilterSize,
						  hybridAttendFeatureMaps,
						  stateDepth,
						  annotationDepth,
						  L)
	parent.__init(self)

	self.scoreDepth = scoreDepth
	self.hybridAttendFilterSize = hybridAttendFilterSize
	self.hybridAttendFeatureMaps = hybridAttendFeatureMaps
	self.stateDepth = stateDepth
	self.annotationDepth = annotationDepth
	self.L = L

	local h 			= nn.Identity()()
	local prev_alpha 	= nn.Identity()()
	local prev_s 		= nn.Identity()()

	------------------ Vh ------------------
	local Vh = nn.TemporalConvolution(annotationDepth,scoreDepth,1)(h)

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
	local padded_alpha = nn.Padding(1,pad_right,2)(nn.Padding(1,-pad_left,2)(prev_alpha))
	local F = nn.TemporalConvolution(1,hybridAttendFeatureMaps,hybridAttendFilterSize)(padded_alpha)
	local UF = nn.TemporalConvolution(hybridAttendFeatureMaps,scoreDepth,1)(F)

	------------------ Ws ------------------
	local ws = nn.TemporalConvolution(1,scoreDepth,stateDepth)(prev_s)
	local Ws = nn.Reshape(L,scoreDepth)(nn.Replicate(L,1,2)(ws))
	
	------------------ tanh ------------------
	local Z = nn.CAddTable()({Ws,Vh,UF})
	local tanh = nn.Tanh()(nn.AddBias(scoreDepth,false)(Z))

	------------------ e_t ------------------
	local e = nn.TemporalConvolution(scoreDepth,1,1)(tanh)

	------------------ alpha_t ------------------
	local alpha = nn.SoftMax()(nn.Reshape(L)(e))

	------------------ c_t ------------------
	local c = nn.MM()({nn.Reshape(1,L)(alpha),h)
	

