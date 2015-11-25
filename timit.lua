require 'nn';
require 'Attention';
require 'LSTM';
require 'nngraph';
require 'RNN';
require 'hdf5';
require 'xlua';
require 'optim';
require 'cunn';

cutorch.setDevice(2)
savedir = '/scratch/jmj/timit/'

------------------ Data ------------------
local file = hdf5.open('/scratch/jmj/timit/cqt.h5')
data = file:all()
file:close()

numPhonemes = 62
data.train.y = data.train.y + 1
data.valid.y = data.valid.y + 1
data.test.y  = data.test.y + 1

--[[
maxNumSamples = 20
data.train.x = data.train.x[{{1,maxNumSamples}}]
data.train.y = data.train.y[{{1,maxNumSamples}}]+1
data.valid.x = data.valid.x[{{1,maxNumSamples}}]
data.valid.y = data.valid.y[{{1,maxNumSamples}}]+1
]]
------------------ Encoder ------------------
seqLength       = data.train.x:size(2)
inputFrameSize  = data.train.x:size(3)
outputFrameSize = 100
kW = 5
enc_inp = nn.Identity()()
convlayer = nn.Sequential()
convlayer:add(nn.TemporalConvolution(inputFrameSize,outputFrameSize,kW))
convlayer:add(nn.ReLU())
convlayer:add(nn.TemporalMaxPooling(2,2))
L = (seqLength - kW + 1)/2
fRNN = nn.RNN(nn.LSTM(outputFrameSize,outputFrameSize,False),L,false)(convlayer(enc_inp))
bRNN = nn.RNN(nn.LSTM(outputFrameSize,outputFrameSize,False),L,true)(convlayer(enc_inp))
concat = nn.JoinTable(2,2)({fRNN,bRNN})
encoder = nn.gModule({enc_inp},{concat})

------------------ Decoder ------------------
scoreDepth = 30
hybridAttendFilterSize = 5
hybridAttendFeatureMaps = 20
stateDepth = 30
annotationDepth = outputFrameSize*2
outputDepth = numPhonemes
T = data.train.y:size(2)
peepholes = false
mlpWidth = 100

decoder_recurrent = nn.LSTM(stateDepth,stateDepth,peepholes)
dec_mlp_inp = nn.Identity()()
mlp_inp = nn.JoinTable(1,1)(dec_mlp_inp)
mlp     = nn.Sequential()
mlp:add(nn.Linear(stateDepth+annotationDepth,mlpWidth))
mlp:add(nn.ReLU())
mlp:add(nn.Linear(mlpWidth,outputDepth))
mlp:add(nn.LogSoftMax())
decoder_mlp = nn.gModule({dec_mlp_inp},{mlp(mlp_inp)})
decoder = nn.Attention(decoder_recurrent,
                       decoder_mlp,
					   scoreDepth,
					   hybridAttendFilterSize,
					   hybridAttendFeatureMaps,
					   stateDepth,
					   annotationDepth,
					   outputDepth,
					   L,
					   T)

------------------ Autoencoder ------------------
autoenc_inp           = nn.Identity()()
autoencoder           = nn.gModule({autoenc_inp},{decoder(encoder(autoenc_inp))})

autoencoder           = autoencoder:cuda()

parameters, gradients = autoencoder:getParameters()

------------------ Train ------------------
batchSize = 100
optimMethod = optim.adam
function train()
	local numSamples     = data.train.x:size(1)
	local shuffle        = torch.randperm(numSamples):long()
	local NLL            = 0
	local numCorrect     = 0
	local numPredictions = 0
	for t=1,numSamples,batchSize do
		xlua.progress(t+batchSize-1,numSamples)
		local indices  = shuffle[{{ t, math.min(t+batchSize-1,numSamples) }}]
		local batchX   = data.train.x:index(1,indices):cuda()
		local batchY   = data.train.y:index(1,indices):cuda()
		local optimfunc = function(x)
			if x ~= parameters then
				parameters:copy(x)
			end
			autoencoder:zeroGradParameters()
			local logprobs  = autoencoder:forward(batchX)
			local labelmask = torch.zeros(batchY:size(1),T,outputDepth):scatter(3,batchY:view(batchY:size(1),T,1):long(),1):cuda()			
			local batchNLL  = -torch.cmul(labelmask,logprobs):sum()
			NLL             = NLL + batchNLL
			batchNLL        = batchNLL/batchSize
			local dLdlogp   = -labelmask/batchSize
			autoencoder:backward(batchX,dLdlogp)
			
			local _, pred   = logprobs:max(3)
			pred = pred:squeeze()
			numCorrect      = numCorrect + torch.eq(pred,batchY:cuda()):sum()
			numPredictions  = numPredictions + batchY:nElement()
			return batchNLL, gradients
		end
		optimMethod(optimfunc, parameters, optimState)
	end
	local accuracy = numCorrect/numPredictions
	NLL            = NLL/numSamples
	return accuracy, NLL
end

------------------ Evaluate ------------------
function evaluate(data)
	local numSamples     = data.x:size(1)
	local NLL            = 0
	local numCorrect     = 0
	local numPredictions = 0
	for t=1,numSamples,batchSize do
		xlua.progress(t+batchSize-1,numSamples)
		local indices   = {{ t, math.min(t+batchSize-1,numSamples) }}
		local batchX    = data.x[indices]:cuda()
		local batchY    = data.y[indices]:cuda()
		local logprobs  = autoencoder:forward(batchX)
		local labelmask = torch.zeros(batchSize,T,outputDepth):scatter(3,batchY:reshape(batchSize,T,1):long(),1):cuda()			
		local batchNLL  = -torch.cmul(labelmask,logprobs):sum()
		NLL             = NLL + batchNLL
		local _, pred   = logprobs:max(3)
		pred = pred:squeeze()
		numCorrect      = numCorrect + torch.eq(pred,batchY):sum()
		numPredictions  = numPredictions + batchY:nElement()
	end
	local accuracy = numCorrect/numPredictions
	NLL            = NLL/numSamples
	return accuracy, NLL
end

------------------ Run ------------------
function updateLog(log,accuracy,nll)
	if log then
		log.accuracy = torch.cat(log.accuracy,torch.Tensor(1):fill(accuracy),1)
		log.nll      = torch.cat(log.nll,torch.Tensor(1):fill(nll),1)
	else
		log = {}
		log.accuracy = torch.Tensor(1):fill(accuracy)
		log.nll      = torch.Tensor(1):fill(nll)
	end
	return log
end
	
numEpochs = 100
local trainLog, validLog
for epoch = 1, numEpochs do
	print('---------- epoch ' .. epoch .. '----------')
	local start = sys.clock()
	local trainAccuracy, trainNLL = train()
	local time = 
	print('training time   =', torch.round(10*(sys.clock()-start)/60)/10 .. ' minutes')
	trainLog = updateLog(trainLog, trainAccuracy, trainNLL)
	print('train Accuracy  =', torch.round(100*100*trainAccuracy)/100 .. '%')
	print('train NLL       =', torch.round(100*trainNLL)/100)
	
	local start = sys.clock()
	local validAccuracy, validNLL = evaluate(data.valid)
	print('validation time =', torch.round(10*(sys.clock()-start)/60)/10 .. ' minutes')
	validLog = updateLog(validLog, validAccuracy, validNLL)
	print('valid Accuracy  =', torch.round(100*100*validAccuracy)/100 .. '%')
	print('valid NLL       =', torch.round(100*validNLL)/100)
	
	local writeFile = hdf5.open(paths.concat(savedir,'log.h5'),'w')
	writeFile:write('train',trainLog)
	writeFile:write('valid',validLog)
	writeFile:close()
end
