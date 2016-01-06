require 'nn';

local AddDim, parent = torch.class('nn.AddDim', 'nn.Module')

function AddDim:__init(dim,ndim)
	parent.__init(self)

	assert(dim ~= nil, 'dim cannot be nil')
	if ndim ~= nil then
		assert(dim <= ndim, 'dim must be <= ndim')
	end
	self.dim = dim
	self.ndim = ndim
end

function AddDim:updateOutput(input)
	local dim
	if input:nDimension() == self.ndim then
		-- nonbatch mode
		dim = self.dim
	elseif input:nDimension() == self.ndim + 1 then
		-- batch mode
		dim = self.dim+1
	else
		error('inconsistent tensor size')
	end
	local size = {}
	if dim == 0 then
		table.insert(size,1)
	end
	for i=1,#input:size() do
		table.insert(size,input:size(i))
		if i == dim then
			table.insert(size,1)
		end
	end
	self.output = torch.view(input,unpack(size))
	return self.output
end

function AddDim:updateGradInput(input, gradOutput)
	self.gradInput = torch.view(gradOutput,input:size())
	return self.gradInput
end
