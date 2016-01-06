require 'nn';

local MonotonicAlignment, parent = torch.class('nn.MonotonicAlignment', 'nn.Module')

function MonotonicAlignment:__init(lambda)
   parent.__init(self)

   self.lambda = lambda or 1
   self.output = torch.Tensor()
   self.gradInput = {torch.Tensor(),torch.Tensor()}

   self.cumsum_alpha = torch.Tensor()
   self.cumsum_prev  = torch.Tensor()
   self.penalty = torch.Tensor()
   self.indices = torch.Tensor()
   self.gradDiff = torch.Tensor()
end

function MonotonicAlignment:updateOutput(input)
	assert(type(input) == 'table', 'input must be a table with {alpha, prev_alpha}')

	local alpha, prev_alpha = unpack(input)

	if alpha:nDimension() == 1 then
		-- non-batch mode
		assert(alpha:size(1) == prev_alpha:size(1), 'alpha and prev_alpha must be the same size')
		self.cumsum_alpha:cumsum(alpha)
		self.cumsum_prev:cumsum(prev_alpha)
		self.penalty:cmax(torch.sum(self.cumsum_alpha-self.cumsum_prev,1),0)
	elseif alpha:nDimension() == 2 then
		-- batch mode
		assert(alpha:size(1) == prev_alpha:size(1), 'alpha and prev_alpha must be the same size')
		assert(alpha:size(2) == prev_alpha:size(2), 'alpha and prev_alpha must be the same size')
		self.cumsum_alpha:cumsum(alpha,2)
		self.cumsum_prev:cumsum(prev_alpha,2)
		self.penalty:cmax(torch.sum(self.cumsum_alpha-self.cumsum_prev,2),0)
	else
		error('alpha must have less than or equal to 2 dimensions')
	end
	self.penalty:mul(self.lambda)
	self.output = alpha
	return self.output
end

function MonotonicAlignment:updateGradInput(input,gradOutput)
	assert(type(input) == 'table', 'input must be a table with {alpha, prev_alpha}')

	local alpha, prev_alpha = unpack(input)

	local penalty_ind = torch.gt(self.penalty,0):float():type(alpha:type())
	local gradDiff
	if alpha:nDimension() == 1 then
		-- non-batch mode
		assert(alpha:size(1) == prev_alpha:size(1), 'alpha and prev_alpha must be the same size')
		local L = alpha:size(1)
		self.indices:resizeAs(alpha):fill(1)
		self.indices:cumsum(self.indices)
		self.gradDiff = (self.indices:mul(-1):add(L+1)):mul(penalty_ind[1])
	elseif alpha:nDimension() == 2 then
		-- batch mode
		assert(alpha:size(1) == prev_alpha:size(1), 'alpha and prev_alpha must be the same size')
		assert(alpha:size(2) == prev_alpha:size(2), 'alpha and prev_alpha must be the same size')
		local L = alpha:size(2)
		local batchSize = alpha:size(1)
		local penalty_ind = penalty_ind:view(batchSize,1):expand(alpha:size())
		self.indices:resizeAs(alpha):fill(1)
		self.indices:cumsum(self.indices,2)
		self.gradDiff = (self.indices:mul(-1):add(L+1)):cmul(penalty_ind)
	else
		error('alpha must have less than or equal to 2 dimensions')
	end
	self.gradDiff:mul(self.lambda)
	self.gradInput[1]:resizeAs(alpha):copy(gradOutput)
	self.gradInput[2]:resizeAs(prev_alpha):zero()
	self.gradInput[1]:add(self.gradDiff)
	self.gradInput[2]:add(-self.gradDiff)
	return self.gradInput
end

