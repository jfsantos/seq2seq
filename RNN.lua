require 'nn'

local RNN, parent = torch.class('nn.RNN','nn.Module')

function RNN:__init(recurrent,T,reverse)
	parent.__init(self)

	assert(recurrent ~= nil, "recurrent cannot be nil")
	assert(T ~= nil, "length of sequence must be specified")
	assert(recurrent.dimoutput ~= nil, "recurrent must specify dimoutput")

	self.recurrent = recurrent
	self.dimoutput = recurrent.dimoutput
	self.T = T
	self.reverse = reverse or false

	self.rnn = self:buildClones()
	self:resetCloneParameters();
end

function RNN:buildClones()
	local clones = {}
	local p,dp = self.recurrent:parameters()
	if p == nil then
		p = {}
	end
	local mem = torch.MemoryFile("w"):binary()
	mem:writeObject(self.recurrent)
	for t = 1, self.T do
		local reader = torch.MemoryFile(mem:storage(), "r"):binary()
		local clone = reader:readObject()
		reader:close()
		local cp,cdp = clone:parameters()
		for i=1,#p do
			cp[i]:set(p[i])
			cdp[i]:set(dp[i])
		end
		clones[t] = clone
		collectgarbage()
	end
	mem:close()
	return clones
end

function RNN:resetCloneParameters()
	local p,dp = self.recurrent:parameters()
	if p == nil then
		p = {}
	end
	for t=1,self.T do
		local cp,cdp = self.rnn[t]:parameters()
		for i=1,#p do
			cp[i]:set(p[i])
			cdp[i]:set(dp[i])
		end
	end
	collectgarbage()
	return p,dp
end

function RNN:parameters()
	return self:resetCloneParameters()
end

function RNN:getParameters()
	-- get parameters
	local parameters,gradParameters = self:parameters()
	local p, dp = self.flatten(parameters), self.flatten(gradParameters)
	self:resetCloneParameters();
	return p, dp
end

function RNN:training()
	for t=1,self.T do
		self.rnn[t]:training()
	end
end

function RNN:evaluate()
	for t=1,self.T do
		self.rnn[t]:evaluate()
	end
end

function RNN:float()
	for t=1,self.T do
		self.clone[t] = self.clones[t]:float()
	end
	return self:type('torch.FloatTensor')
end

function RNN:double()
	for t=1,self.T do
		self.clone[t] = self.clones[t]:double()
	end
	return self:type('torch.DoubleTensor')
end

function RNN:cuda()
	for t=1,self.T do
		self.clone[t] = self.clones[t]:cuda()
	end
	return self:type('torch.CudaTensor')
end

function RNN:resetOutput(input)
	if self.sequence_dim == 1 then
		self.output = torch.zeros(self.T, self.dimoutput):type(input:type())
	else
		self.output = torch.zeros(input:size(1), self.T, self.dimoutput):type(input:type())
	end
end

function RNN:updateOutput(input)
	-- 2D input ~ SGD mode
	-- 3D input ~ batch mode
	if input:nDimension() == 2 then
		self.sequence_dim = 1 
	elseif input:nDimension() == 3 then 
		self.sequence_dim = 2
	else
		error('input dimension must be 2D or 3D')
	end

	assert(input:size(self.sequence_dim)==self.T, "sequence size of input must match self.T")
	
	local resetflag = not self.output
					  or  self.output:nDimension()~=input:nDimension()
					  or  self.output:size(1)~=input:size(1)
	if resetflag then
		self:resetOutput(input)
	end

	self.h = {}
	local y, h
	local start,finish,step = 1, self.T, 1
	if self.reverse then
		start,finish,step = self.T, 1, -1
	end
	for t = start,finish,step do
		local x = input:select(self.sequence_dim,t)
		y, h = unpack(self.rnn[t]:forward({x,y,h}))
		self.output:select(self.sequence_dim,t):copy(y)
		self.h[t] = h
	end

	return self.output
end

function RNN:updateGradInput(input, gradOutput)

	assert(input:size(self.sequence_dim)==self.T, "sequence size of input must match self.T")

	if not self.gradInput or self.gradInput:size() ~= input:size() then
		self.gradInput = torch.zeros(input:size()):type(input:type())
	end
	
	self.gradh = {}
	local dEdx,dEdy,dEdpy,dEdph
	local start,finish,step = 1, self.T, 1
	if self.reverse then
		start,finish,step = self.T, 1, -1
	end
	for t = finish,start,-step do
		local x	= input:select(self.sequence_dim,t)
		local y,h
		if t == start then
			y = nil
			h = nil
		else
			y = self.output:select(self.sequence_dim,t-step)
			h = self.h[t-step]
		end
		dEdy = gradOutput:select(self.sequence_dim,t)
		dEdy = dEdy + (dEdpy or 0)
		dEdx,dEdpy,dEdph = unpack(self.rnn[t]:backward({x,y,h},{dEdy,dEdph}))
		self.gradInput:select(self.sequence_dim,t):copy(dEdx)
		self.gradh[t-step] = dEdph
	end

	return self.gradInput
end




