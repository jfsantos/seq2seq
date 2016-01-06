require 'nn';

local ExpandAs, parent = torch.class('nn.ExpandAs', 'nn.Module')

function ExpandAs:__init(dim,ndim)
	parent.__init(self)

	assert(dim ~= nil, 'dim cannot be nil')
	if ndim ~= nil then
		assert(dim <= ndim, 'dim must be <= ndim')
	end
	self.dim = dim
	self.ndim = ndim
	self.zeros = torch.Tensor()
end

function ExpandAs:updateOutput(input)
	local x,y = unpack(input)
	-- replicate x as y
	local xndim = x:nDimension()
	assert(x:nDimension() == y:nDimension(), 'x and y must have equal num dimensions')
	assert((self.ndim==nil) or (xndim==self.ndim) or (xndim==self.ndim+1), 'inconsistent tensor size')
	self.output = torch.expandAs(x,y)
	return self.output
end

function ExpandAs:updateGradInput(input, gradOutput)
	local x,y = unpack(input)
	assert(x:nDimension() == y:nDimension(), 'x and y must have equal num dimensions')
	local xndim = x:nDimension()
	local gradInput = {}
	if (self.ndim==nil) or (xndim==self.ndim) then
		-- nonbatch mode
		gradInput[1] = gradOutput:sum(self.dim)
	elseif xndim==self.ndim+1 then
		-- batch mode
		gradInput[1] = gradOutput:sum(self.dim+1)
	else
		error('inconsistent tensor size')
	end
	gradInput[2] = self.zeros:resizeAs(y):zero()
	self.gradInput = gradInput

	return self.gradInput
end
