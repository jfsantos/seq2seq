require 'nn';

local AddBias, parent = torch.class('nn.AddBias', 'nn.Module')

function AddBias:__init(inputSize,scalar)
   parent.__init(self)
  
   local size = inputSize
   if scalar then size=1 end
   self.scalar = scalar
   self.bias = torch.Tensor(size)
   self.gradBias = torch.Tensor(size)
   
   self._ones = torch.Tensor{1}
   self._bias_grid = torch.Tensor()
   self:reset()
end

function AddBias:reset(stdv)
   if stdv then 
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.bias:size(1))
   end

   self.bias:uniform(-stdv, stdv)
end

function AddBias:updateOutput(input)
   self.output = self.output:type(input:type()):resizeAs(input):copy(input)
   if self.scalar then
      self.output:add(self.bias[1]);
   else
      if input:isSameSizeAs(self.bias) then
         self.output:add(self.bias)
      else
		 if input:nDimension() == 2 then
			 -- single SGD mode
			 local length = input:size(1)
			 if self._ones:size(1) ~= length then
				self._ones:resize(length):fill(1)
			 end
			 local bias = self.bias:view(-1)
			 local output = self.output:view(length, -1)
			 output:addr(1, self._ones, bias)
		 elseif input:nDimension() == 3 then 
			 -- batchMode
			 local length = input:size(2)
			 if self._ones:size(1) ~= length then
				self._ones:resize(length):fill(1)
			 end
			 local bias = self.bias:view(-1)
   			 self._bias_grid = self._bias_grid:type(input:type())
			 self._bias_grid = self._bias_grid:resizeAs(input[1]):fill(0):addr(1, self._ones, bias)
			 self.output:add(self._bias_grid:resize(1,input:size(2),input:size(3)):expandAs(input))
		 else
			 error('input must be a 1D, 2D, or 3D tensor')
		 end

      end
   end
   return self.output
end

function AddBias:updateGradInput(input, gradOutput)
   if self.gradInput then
	  self.gradInput = self.gradInput:type(gradOutput:type())
      self.gradInput:resizeAs(gradOutput):copy(gradOutput) 
      return self.gradInput
   end
end

function AddBias:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if self.gradBias:size(1) == 1 then
      self.gradBias[1] = self.gradBias[1] + scale*gradOutput:sum();
   else
      if input:isSameSizeAs(self.bias) then
         self.gradBias:add(scale, gradOutput)
      else
			if input:nDimension() == 2 then
				-- SGD
				local gradOutput = gradOutput:view(input:size(1), -1)
				self.gradBias:view(-1):addmv(scale, gradOutput:t(), self._ones)
			elseif input:nDimension() == 3 then
				-- batchMode
				self.gradBias:addmv(scale, torch.sum(gradOutput,1):squeeze():t(), self._ones)
			else
				error('input must be 1D, 2D, or 3D tensor')
			end
      end
   end
end
