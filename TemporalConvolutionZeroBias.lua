local TemporalConvolutionZeroBias, parent = torch.class('nn.TemporalConvolutionZeroBias', 'nn.Module')

function TemporalConvolutionZeroBias:__init(inputFrameSize, outputFrameSize, kW, dW)
   parent.__init(self)

   dW = dW or 1

   self.inputFrameSize = inputFrameSize
   self.outputFrameSize = outputFrameSize
   self.kW = kW
   self.dW = dW

   self.weight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.bias = torch.Tensor(outputFrameSize):zero()
   self.gradWeight = torch.Tensor(outputFrameSize, inputFrameSize*kW)
   self.gradBias = torch.Tensor(outputFrameSize)
   
   self:reset()
end

function TemporalConvolutionZeroBias:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.inputFrameSize)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
   else
      self.weight:uniform(-stdv, stdv)
   end
   self.bias:zero()
end

function TemporalConvolutionZeroBias:updateOutput(input)
   self.bias:zero()
   return input.nn.TemporalConvolution_updateOutput(self, input)
end

function TemporalConvolutionZeroBias:updateGradInput(input, gradOutput)
   self.bias:zero()
   if self.gradInput then
      return input.nn.TemporalConvolution_updateGradInput(self, input, gradOutput)
   end
end

function TemporalConvolutionZeroBias:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   input.nn.TemporalConvolution_accGradParameters(self, input, gradOutput, scale)
   self.gradBias:zero()
   self.bias:zero()
end

-- we do not need to accumulate parameters when sharing
TemporalConvolutionZeroBias.sharedAccUpdateGradParameters = TemporalConvolutionZeroBias.accUpdateGradParameters

