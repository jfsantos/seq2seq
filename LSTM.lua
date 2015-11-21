require 'nn';
require 'nngraph';

local LSTM, parent = torch.class('nn.LSTM','nn.Module')

function LSTM:__init(diminput,dimoutput,peepholes)
	parent.__init(self)

	assert(diminput  ~= nil, "diminput must be specified")
	assert(dimoutput ~= nil, "dimoutput must be specified")

	self.diminput  = diminput
	self.dimoutput = dimoutput
	self.peepholes = peepholes or false

	local inp    = nn.Identity()()
	local prev_c = nn.Identity()()
	local prev_h = nn.Identity()()

	self.inp = inp
	self.prev_c = prev_c
	self.prev_h = prev_h

	self.gateinputs = {}
	local function gateInputs(peephole)
		local i2h = nn.Linear(diminput,dimoutput)
		local h2h = nn.Linear(dimoutput,dimoutput)
		table.insert(self.gateinputs,i2h)
		if peephole then
			print('peepholes!')
			local c2h = nn.Linear(dimoutput,dimoutput)
			return nn.CAddTable()({i2h(inp),h2h(prev_h),c2h(peephole)})
		else
			return nn.CAddTable()({i2h(inp),h2h(prev_h)})
		end
	end

	local peephole = nil
	if self.peepholes then
		peephole = prev_c
	end
	local in_gate		= nn.Sigmoid()(gateInputs(peephole))
	local forget_gate	= nn.Sigmoid()(gateInputs(peephole))
	local in2cell		= nn.Tanh()(gateInputs(false))
	local next_c		= nn.CAddTable()({nn.CMulTable()({forget_gate, prev_c}),
										  nn.CMulTable()({in_gate, in2cell})})
	if self.peepholes then
		peephole = next_c
	end
	local out_gate 		= nn.Sigmoid()(gateInputs(peephole))
	local next_h		= nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

	self.in_gate = in_gate
	self.forget_gate = forget_gate
	self.in2cell = in2cell
	self.next_c = next_c

	self.lstm = nn.gModule({inp,prev_h,prev_c},{next_h,next_c})
end

function LSTM:parameters()
	return self.lstm:parameters()
end

function LSTM:training()
	self.lstm:training()
end

function LSTM:evaluate()
	self.lstm:evaluate()
end

function LSTM:float()
	self.lstm 	= self.lstm:float()
	self.zeros	= self.zeros:float()
	return self:type('torch.FloatTensor')
end

function LSTM:double()
	self.lstm 	= self.lstm:double()
	self.zeros	= self.zeros:double()
	return self:type('torch.DoubleTensor')
end

function LSTM:cuda()
	self.lstm 	= self.lstm:cuda()
	self.zeros	= self.zeros:cuda()
	return self:type('torch.CudaTensor')
end

function LSTM:resetZeros(inp)
	if inp:nDimension() == 2 then
		self.zeros = torch.zeros(inp:size(1),self.dimoutput):type(inp:type())
	else
		self.zeros = torch.zeros(self.dimoutput):type(inp:type())
	end
end

function LSTM:updateOutput(input)
	local inp, prev_h, prev_c = unpack(input)

	if not self.zeros then
		self:resetZeros(inp)
	else
		if self.zeros:nDimension()~=inp:nDimension() or self.zeros:size(1)~=inp:size(1) then
			self:resetZeros(inp)
		end
	end

	prev_c = prev_c or self.zeros
	prev_h = prev_h or self.zeros

	self.output = self.lstm:forward({inp,prev_h,prev_c})
	return self.output
end

function LSTM:updateGradInput(input, gradOutput)
	local inp, prev_h, prev_c = unpack(input)
	local dEdh,dEdc = unpack(gradOutput)

	assert(dEdh ~= nil, "dEdh should not be nil")
	assert(prev_c ~= nil or dEdc ~= nil, "prev_c and dEdc cannot both be nil")
	
	prev_c = prev_c or self.zeros
	prev_h = prev_h or self.zeros
	dEdc = dEdc or self.zeros

	--print('LSTM')
	--print('inp',inp)
	--print('prev_h',prev_h)
	--print('prev_c',prev_c)
	local dEdx,dEdph,dEdpc = unpack(self.lstm:backward({inp,prev_h,prev_c},{dEdh,dEdc}))
	self.gradInput = {dEdx,dEdph,dEdpc}
	return self.gradInput
end

