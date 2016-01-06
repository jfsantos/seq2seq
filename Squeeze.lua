require 'nn';

local Squeeze, parent = torch.class('nn.Squeeze', 'nn.Module')

function Squeeze:__init(dim,ndim)
	parent.__init(self)

	assert(dim ~= nil, 'dim cannot be nil')
	if ndim ~= nil then
		assert(dim <= ndim, 'dim must be <= ndim')
	end

	self.dim = dim
	self.ndim = ndim

end

function Squeeze:updateOutput(input)
	if (self.ndim == nil) or (input:nDimension() == self.ndim) then
		-- non-batch mode
		self.output = torch.squeeze(input,self.dim)
	elseif input:nDimension() == self.ndim+1 then
		-- batch mode
		self.output = torch.squeeze(input,self.dim+1)
	else
		print(input:size())
		error('inconsistent tensor size, self.ndim = ' .. self.ndim)
	end
	return self.output
end

function Squeeze:updateGradInput(input, gradOutput)
	self.gradInput = torch.viewAs(gradOutput,input)
	return self.gradInput
end
