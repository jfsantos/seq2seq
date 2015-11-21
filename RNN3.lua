require 'nn'

local RNN3, parent = torch.class('nn.RNN3','nn.Module')

function RNN3:__init(recurrent,T,dimoutput,reverse)
	parent.__init(self)

	assert(recurrent ~= nil, "recurrent cannot be nil")
	assert(T ~= nil, "length of sequence must be specified")
	assert(dimoutput ~= nil, "recurrent must specify dimoutput")

	self.recurrent = recurrent
	self.dimoutput = dimoutput
	self.T = T
	self.reverse = reverse or false
	
	self.output = torch.Tensor(T,dimoutput)

	self.rnn = self:buildClones()
	self:resetCloneParameters();
end

function RNN3:buildClones()
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

function RNN3:resetCloneParameters()
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

function RNN3:parameters()
	return self:resetCloneParameters()
end

function RNN3:getParameters()
	-- get parameters
	local parameters,gradParameters = self:parameters()
	local p, dp = self.flatten(parameters), self.flatten(gradParameters)
	self:resetCloneParameters();
	return p, dp
end

function RNN3:training()
	for t=1,self.T do
		self.rnn[t]:training()
	end
end

function RNN3:evaluate()
	for t=1,self.T do
		self.rnn[t]:evaluate()
	end
end

function RNN3:float()
	for t=1,self.T do
		self.clone[t] = self.clones[t]:float()
	end
	return self:type('torch.FloatTensor')
end

function RNN3:double()
	for t=1,self.T do
		self.clone[t] = self.clones[t]:double()
	end
	return self:type('torch.DoubleTensor')
end

function RNN3:cuda()
	for t=1,self.T do
		self.clone[t] = self.clones[t]:cuda()
	end
	return self:type('torch.CudaTensor')
end


local function getBatchSize(input)
	if type(input) == 'table' then
		return getBatchSize(input[1])
	else
		if input:nDimension() == 2 then
			return  0  -- SGD (non-batch)
		else
			return input:size(1)  -- batch mode
		end
	end
end


function RNN3:updateOutput(input)
	--print('RNN3 type(input) ==',type(input))
	--print('RNN3 input = ',input)
	local batchSize = getBatchSize(input)
	
	if batchSize == 0 then
		self.sequence_dim = 1
		self.output:resize(self.T,self.dimoutput)
	else
		self.sequence_dim = 2
		self.output:resize(batchSize,self.T,self.dimoutput)
	end
	
	self.h = {}
	local y, h
	local start,finish,step = 1, self.T, 1
	if self.reverse then
		start,finish,step = self.T, 1, -1
	end
	for t = start,finish,step do
		--print('==================================> RNN t =',t)
		y, h = unpack(self.rnn[t]:forward({input,y,h}))
		--print('RNN3 output')
		--print('y',y)
		--print('h',h)
		self.output:select(self.sequence_dim,t):copy(y)
		self.h[t] = h
	end

	return self.output
end

function RNN3:_resizeGradInput(input,gradInput)
	if type(input) == 'table' then
		if type(gradInput) ~= 'table' then
			gradInput = {}
		end
		for i = 1, #input do
			gradInput[i] = self:_resizeGradInput(input[i],gradInput[i])
		end
	else
		gradInput = gradInput or input.new()
		gradInput:resizeAs(input):zero()
	end
	return gradInput
end

function RNN3:updateGradInput(input, gradOutput)
	local batchSize = getBatchSize(input)
	
	self.gradInput = self:_resizeGradInput(input,self.gradInput)

	local function updateGradInput(dEdx,gradInput)
		if type(gradInput) == 'table' then
			for i = 1, #gradInput do
				updateGradInput(dEdx[i],gradInput[i])
			end
		else
			gradInput:add(dEdx)
		end
	end

	self.gradh = {}
	local dEdx	 -- gradient of loss w.r.t. inputs
	local dEdy   -- gradient of loss w.r.t. output
	local dEdpy  -- gradient of loss w.r.t. prev output
	local dEdh   -- gradient of loss w.r.t. hidden state
	local dEdph  -- gradient of loss w.r.t. prev hidden state
	local start,finish,step = 1, self.T, 1
	if self.reverse then
		start,finish,step = self.T, 1, -1
	end
	for t = finish,start,-step do
		local x = input
		local y,h
		if t == start then
			y = nil
			h = nil
		else
			-- note: self.sequence_dim is set in updateOutput
			y = self.output:select(self.sequence_dim,t-step)
			h = self.h[t-step]
		end
		dEdy = gradOutput:select(self.sequence_dim,t)
		dEdy = dEdy + (dEdpy or 0)
		dEdh = dEdph
		dEdx,dEdpy,dEdph = unpack(self.rnn[t]:backward({x,y,h},{dEdy,dEdh}))
		updateGradInput(dEdx,self.gradInput)
		self.gradh[t-step] = dEdph
	end

	return self.gradInput 
end
