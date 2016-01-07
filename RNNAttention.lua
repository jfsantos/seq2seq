require 'nn'

local RNNAttention, parent = torch.class('nn.RNNAttention','nn.Module')

function RNNAttention:__init(recurrent,dimoutput,reverse)
	parent.__init(self)

	assert(recurrent ~= nil, "recurrent cannot be nil")
	assert(dimoutput ~= nil, "recurrent must specify dimoutput")

	self.recurrent = recurrent
	self.dimoutput = dimoutput
	self.reverse = reverse or false
	
	self.output = torch.Tensor()
	self.rnn = {}

	self.zeros_y = torch.Tensor()
end

function RNNAttention:addClone()
	local p,dp = self.recurrent:parameters()
	if p == nil then
		p = {}
	end
	local mem = torch.MemoryFile("w"):binary()
	mem:writeObject(self.recurrent)
	local reader = torch.MemoryFile(mem:storage(), "r"):binary()
	local clone = reader:readObject()
	reader:close()
	local cp,cdp = clone:parameters()
	for i=1,#p do
		cp[i]:set(p[i])
		cdp[i]:set(dp[i])
	end
	if not self.rnn then
		self.rnn = {}
	end
	self.rnn[#self.rnn+1] = clone
	collectgarbage()
	mem:close()
	self:resetCloneParameters()
end

function RNNAttention:resetCloneParameters()
	local p,dp = self.recurrent:parameters()
	if p == nil then
		p = {}
	end
	for t=1,#self.rnn do
		local cp,cdp = self.rnn[t]:parameters()
		for i=1,#p do
			cp[i]:set(p[i])
			cdp[i]:set(dp[i])
		end
	end
	collectgarbage()
	return p,dp
end

function RNNAttention:parameters()
	return self:resetCloneParameters()
end

function RNNAttention:getParameters()
	-- get parameters
	local parameters,gradParameters = self:parameters()
	local p, dp = self.flatten(parameters), self.flatten(gradParameters)
	self:resetCloneParameters();
	return p, dp
end

function RNNAttention:training()
	self.recurrent:training()
	for t=1,#self.rnn do
		self.rnn[t]:training()
	end
end

function RNNAttention:evaluate()
	self.recurrent:evaluate()
	for t=1,#self.rnn do
		self.rnn[t]:evaluate()
	end
end

function RNNAttention:float()
	self.recurrent = self.recurrent:float()
	for t=1,#self.rnn do
		self.rnn[t] = self.rnn[t]:float()
	end
	return self:type('torch.FloatTensor')
end

function RNNAttention:double()
	self.recurrent = self.recurrent:double()
	for t=1,#self.rnn do
		self.rnn[t] = self.rnn[t]:double()
	end
	return self:type('torch.DoubleTensor')
end

function RNNAttention:cuda()
	self.recurrent = self.recurrent:cuda()
	for t=1,#self.rnn do
		self.rnn[t] = self.rnn[t]:cuda()
	end
	return self:type('torch.CudaTensor')
end


local function getBatchSize(input)
	-- assume 2D = nonbatch, 3d = batch
	-- if input is a table, then assume
	-- input[1] is a template
	--
	-- returns batchSize
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


function RNNAttention:setT(T)
	self.T = T
end

function RNNAttention:apply2clones(func)
	func(self.recurrent)
	for t = 1, #self.rnn do
		func(self.rnn[t])
	end
end

function RNNAttention:updateOutput(input)
	local T = self.T
	assert(T ~= nil, 'T cannot be nil')

	local batchSize = getBatchSize(input)
	local nonrecurrent, y = unpack(input)
	
	self.batchSize = batchSize
	if batchSize == 0 then
		self.sequence_dim = 1
		self.output:resize(T,self.dimoutput)
		self.zeros_y:resize(self.dimoutput)
	else
		self.sequence_dim = 2
		self.output:resize(batchSize,T,self.dimoutput)
		self.zeros_y:resize(batchSize,self.dimoutput)
	end
	
	self.h = {}
	local prev_y, next_y, h
	local start,finish,step = 1, T, 1
	if self.reverse then
		start,finish,step = T, 1, -1
	end
	for t = start,finish,step do
		if not self.rnn[t] then
			self:addClone()
		end
		if t == start then
			prev_y = self.zeros_y
		else
			prev_y = y:select(self.sequence_dim,t-step)
		end
		local x = {nonrecurrent, prev_y}
		next_y, h = unpack(self.rnn[t]:forward({x,h}))
		self.output:select(self.sequence_dim,t):copy(next_y)
		self.h[t] = h
	end

	return self.output
end


function RNNAttention:_resizeGradInput(input,gradInput)
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

function RNNAttention:updateGradInput(input, gradOutput)
	local batchSize = getBatchSize(input)
	local T = self.T
	self.gradInput = self:_resizeGradInput(input,self.gradInput)

	local function updateGradInput(dEdx,gradInput,t)
		if type(gradInput) == 'table' then
			for i = 1, #gradInput do
				updateGradInput(dEdx[i],gradInput[i],t)
			end
		else
			if t then
				gradInput:select(self.sequence_dim,t):add(dEdx)
			else
				gradInput:add(dEdx)
			end
		end
	end

	self.gradh = {}
	local dEdx	 -- gradient of loss w.r.t. inputs
	local dEdy   -- gradient of loss w.r.t. output
	local dEdh   -- gradient of loss w.r.t. hidden state
	local dEdph  -- gradient of loss w.r.t. prev hidden state
	local nonrecurrent, y = unpack(input)
	local prev_y, next_y, h
	local start,finish,step = 1, T, 1
	if self.reverse then
		start,finish,step = T, 1, -1
	end
	for t = finish,start,-step do
		local h
		if t == start then
			prev_y = self.zeros_y
			h = nil
		else
			-- note: self.sequence_dim is set in updateOutput
			prev_y = y:select(self.sequence_dim,t-step)
			h = self.h[t-step]
		end
		local x = {nonrecurrent, prev_y}
		dEdy = gradOutput:select(self.sequence_dim,t)
		dEdh = dEdph
		dEdx,dEdph = unpack(self.rnn[t]:backward({x,h},{dEdy,dEdh}))
		updateGradInput(dEdx[1],self.gradInput[1])   -- nonrecurrent input
		updateGradInput(dEdx[2],self.gradInput[2],t) -- y
		self.gradh[t-step] = dEdph
	end

	return self.gradInput 
end
