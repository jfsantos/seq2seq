require 'nn';
require 'utils';

local Recurrent, parent = torch.class('nn.Recurrent','nn.Module')

function Recurrent:__init(recurrent,dimhidden,dimoutput)
	parent.__init(self)
	
	self.recurrent = recurrent
	self.dimhidden = dimhidden
	self.dimoutput = dimoutput

	self.zeros_hidden = self:_recursiveOperation(dimhidden,function(x) return torch.zeros(x) end)	
end

function Recurrent:parameters()
    return self.recurrent:parameters()
end

function Recurrent:training()
    self.recurrent:training()
end

function Recurrent:evaluate()
    self.recurrent:evaluate()
end

function Recurrent:_recursiveOperation(x,func,y)
	local output
	local y = y or {}
	if type(x) == 'table' then
		output = {}
		for i = 1, #x do
			output[i] = self:_recursiveOperation(x[i],func,y[i])
		end
	else
		output = func(x,y)
	end
	return output
end

function Recurrent:float()
	local func = function(x) x = x:float() end
	self:_recursiveOperation(self.recurrent,func)
	self:_recursiveOperation(self.zeros_hidden,func)
    return self:type('torch.FloatTensor')
end

function Recurrent:double()
	local func = function(x) x = x:double() end
	self:_recursiveOperation(self.recurrent,func)
	self:_recursiveOperation(self.zeros_hidden,func)
    return self:type('torch.DoubleTensor')
end

function Recurrent:cuda()
	local func = function(x) x = x:cuda() end
	self:_recursiveOperation(self.recurrent,func)
	self:_recursiveOperation(self.zeros_hidden,func)
    return self:type('torch.CudaTensor')
end

local function getBatchSize(x)
	if type(x) == 'table' then
		return getBatchSize(x[1])
	else
		if x:nDimension() == 3 then
			return x:size(1)   -- batch mode
		else
			return 0           -- SGD
		end
	end
end

function Recurrent:resetZeros(inp)
	local batchSize = getBatchSize(inp)
	local func
    if batchSize > 0 then
		-- batch mode
		func = function(x,y) 
			local dim = y
			if x:nDimension() ~= 2 or x:size(1) ~= batchSize or x:size(2) ~= dim then
				x:resize(batchSize,dim):zero()
			end
			return x
		end
    else
		-- SGD mode
		func = function(x,y) 
			local dim = y
			if x:nDimension() ~= 1 or x:size(1) ~= dim then 
				x:resize(dim):zero()
			end
			return x
		end
	end
	self.zeros_hidden = self:_recursiveOperation(self.zeros_hidden,func,self.dimhidden)
end

function Recurrent:updateOutput(input)
    local inp, prev_h = unpack(input)

	if #inp == 1 then
		inp = inp[1]
	end
	self:resetZeros(inp)

    prev_h = prev_h or self.zeros_hidden

    self.output = self.recurrent:forward({inp,prev_h})
    return self.output
end

function Recurrent:updateGradInput(input, gradOutput)
    local inp, prev_h = unpack(input) 
    local dEdy,dEdh = unpack(gradOutput)

    assert(dEdy ~= nil, "dEdy should not be nil")
    assert(prev_h ~= nil or dEdh ~= nil, "prev_c and dEdc cannot both be nil")

	if #inp == 1 then
		inp = inp[1]
	end
    prev_h = prev_h or self.zeros_hidden
    dEdh = dEdh or self.zeros_hidden

    local dEdx,dEdph = unpack(self.recurrent:backward({inp,prev_h},{dEdy,dEdh}))
    self.gradInput = {dEdx,dEdph}
    return self.gradInput
end

