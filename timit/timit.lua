require 'nn';
package.path = package.path .. ';../?.lua'
require 'Attention';
require 'LSTM';
require 'nngraph';
require 'RNN';
require 'hdf5';
require 'xlua';
require 'optim';
require 'cunn';

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Options')
cmd:option('-device',1,'gpu device')
cmd:option('-savedir','TIMIT/output/logmel','output directory')
cmd:option('-data','TIMIT/logmel.h5','data file')
cmd:option('-maxnorm',1,'max norm for gradient clipping')
cmd:option('-maxnumsamples',1e20,'use small number of samples for code testing')
cmd:option('-batchSize',1,'batch size to use during training')
cmd:option('-weightDecay',1e-4,'weight decay')
cmd:text()

opt = cmd:parse(arg)

cutorch.setDevice(opt.device)
savedir     = opt.savedir
batchSize   = opt.batchSize
maxnorm     = opt.maxnorm
weightDecay = opt.weightDecay

------------------ Data ------------------
local file = hdf5.open(opt.data)
data = file:all()
file:close()

numPhonemes = 62

function processData(data)
    if type(data) == 'table' then
		if batchSize == 1 then
			print('resetting batchSize to 1 since data is a table, which is assumed to have variable length sequences')
			batchSize = 1
		end
        dataset   = {}
        dataset.x = {}
        dataset.y = {}
        i=0
        for k,f in pairs(data) do
            i=i+1
            dataset.x[i] = f.x:cuda()
            dataset.y[i] = f.y:cuda()
            if i >= opt.maxnumsamples then
                break
            end
        end
        dataset.numSamples = i
        return dataset
    else
		data.numSamples = data.x:size(1)
        return data
	end
end

train = processData(data.train)
valid = processData(data.valid)
test  = processData(data.test)

------------------ Encoder ------------------
seqLength       = train.x[1]:size(1)
inputFrameSize  = train.x[1]:size(2)
hiddenFrameSize = 512
outputFrameSize = 256
kW = 5
enc_inp = nn.Identity()()
convlayer = nn.Sequential()
convlayer:add(nn.TemporalConvolution(inputFrameSize,hiddenFrameSize,kW))
convlayer:add(nn.ReLU())
convlayer:add(nn.TemporalMaxPooling(2,2))
L = (seqLength - kW + 1)/2
convlayer:add(nn.TemporalConvolution(hiddenFrameSize,hiddenFrameSize,kW))
convlayer:add(nn.ReLU())
convlayer:add(nn.TemporalMaxPooling(2,2))
L = (L - kW + 1)/2
fRNN = nn.RNN(nn.LSTM(hiddenFrameSize,outputFrameSize,False),L,false)(convlayer(enc_inp))
bRNN = nn.RNN(nn.LSTM(hiddenFrameSize,outputFrameSize,False),L,true)(convlayer(enc_inp))
concat = nn.JoinTable(2,2)({fRNN,bRNN})
encoder = nn.gModule({enc_inp},{concat})

------------------ Decoder ------------------
scoreDepth = 100
hybridAttendFilterSize = 5
hybridAttendFeatureMaps = 128
stateDepth = 200
annotationDepth = outputFrameSize*2
outputDepth = numPhonemes
peepholes = true
mlpWidth = 150 

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
					   L)

------------------ Autoencoder ------------------
autoenc_inp           = nn.Identity()()
autoencoder           = nn.gModule({autoenc_inp},{decoder(encoder(autoenc_inp))})

autoencoder           = autoencoder:cuda()

parameters, gradients = autoencoder:getParameters()

------------------ Train ------------------
optimMethod = optim.adadelta
optimConfig = {
	eps = 1e-8,
	rho = 0.95
}
optimState  = nil
gradnoise   = {
	eta   = 1e-3,
	gamma = 0.55,
	t     = 0
}
function Train()
	local numSamples     = train.numSamples
	local shuffle        = torch.randperm(numSamples):long()
	local NLL            = 0
	local numCorrect     = 0
	local numPredictions = 0
	autoencoder:training()
	for t=1,numSamples,batchSize do
		collectgarbage()
		xlua.progress(t+batchSize-1,numSamples)
		local index      = shuffle[t]
		local X          = train.x[index]
		local Y          = train.y[index]
		local T          = Y:size(1)
		local optimfunc = function(x)
			if x ~= parameters then
				parameters:copy(x)
			end
			autoencoder:zeroGradParameters()
			decoder:setT(T)
			local logprobs  = autoencoder:forward(X)
			local labelmask = torch.zeros(T,outputDepth):cuda():scatter(2,Y:view(T,1),1)
			local nll       = -torch.cmul(labelmask,logprobs):sum()
			NLL             = NLL + nll
			local dLdlogp   = -labelmask
			autoencoder:backward(X,dLdlogp)
			
			local _, pred   = logprobs:max(2)
			pred = pred:squeeze()
			numCorrect      = numCorrect + torch.eq(pred,Y):sum()
			numPredictions  = numPredictions + Y:nElement()

			local gradnorm  = gradients:norm()
			if gradnorm > maxnorm then
				gradients:mul(maxnorm/gradnorm)
			end

			if weightDecay > 0 then
				gradients:add(parameters*weightDecay)
			end
			gradnoise.t = gradnoise.t + 1
			local sigma     = (gradnoise.eta/(1+gradnoise.t)^gradnoise.gamma)^0.5
			gradients:add(torch.randn(gradients:size()):cuda()*sigma)
			return nll, gradients
		end
		optimMethod(optimfunc, parameters, optimConfig, optimState)
	end
	local accuracy = numCorrect/numPredictions
	NLL            = NLL/numSamples
	return accuracy, NLL
end

------------------ Evaluate ------------------
function Evaluate(data)
	local numSamples     = data.numSamples
	local NLL            = 0
	local numCorrect     = 0
	local numPredictions = 0
	autoencoder:evaluate()
	for t=1,numSamples,batchSize do
		xlua.progress(t+batchSize-1,numSamples)
		local index      = t
		local X          = data.x[index]
		local Y          = data.y[index]
		local T          = Y:size(1)
		decoder:setT(T)
		local logprobs   = autoencoder:forward(X)
		local labelmask  = torch.zeros(T,outputDepth):cuda():scatter(2,Y:view(T,1),1)
		local nll        = -torch.cmul(labelmask,logprobs):sum()
		NLL              = NLL + nll
		local _, pred    = logprobs:max(2)
		pred = pred:squeeze()
		numCorrect       = numCorrect + torch.eq(pred,Y):sum()
		numPredictions   = numPredictions + Y:nElement()
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
	local trainAccuracy, trainNLL = Train()
	local time = 
	print('\ntraining time   =', torch.round(10*(sys.clock()-start)/60)/10 .. ' minutes')
	trainLog = updateLog(trainLog, trainAccuracy, trainNLL)
	print('train Accuracy  =', torch.round(100*100*trainAccuracy)/100 .. '%')
	print('train NLL       =', torch.round(100*trainNLL)/100)
	print('||grad||        =', gradients:norm())
	local alpha_train = decoder:alpha():float()
	
	local start = sys.clock()
	local validAccuracy, validNLL = Evaluate(valid)
	print('\nvalidation time =', torch.round(10*(sys.clock()-start)/60)/10 .. ' minutes')
	validLog = updateLog(validLog, validAccuracy, validNLL)
	print('valid Accuracy  =', torch.round(100*100*validAccuracy)/100 .. '%')
	print('valid NLL       =', torch.round(100*validNLL)/100)
	local alpha_valid = decoder:alpha():float()
	
	print('saving to ' .. savedir)
	sys.execute('mkdir -p ' .. savedir)
	local writeFile = hdf5.open(paths.concat(savedir,'log.h5'),'w')
	writeFile:write('train',trainLog)
	writeFile:write('valid',validLog)
	writeFile:write('alpha_train',alpha_train)
	writeFile:write('alpha_valid',alpha_valid)
	writeFile:close()
end
