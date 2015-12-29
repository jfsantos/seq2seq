import librosa
import numpy as np
import os
import h5py
import pickle
from optparse import OptionParser

parser = OptionParser()
parser.add_option('--root',help='root TIMIT dir',dest='root',default='TIMIT')
parser.add_option('--save',help='save directory',dest='save',default='TIMIT')
parser.add_option('--valid',help='validation speakerIDs',dest='valid',default='valid_spkrid.txt')
parser.add_option('--samelenfeat',help='if True, all features will have same length',dest='samelenfeat',default=True)
parser.add_option('--samelenlabs',help='if True, all labels will have same length',dest='samelenlabs',default=False)
(options, args) = parser.parse_args()

featuresSameLength = options.samelenfeat
phonemesSameLength = options.samelenlabs
rootdir            = options.root
savedir            = options.save
rootdirtrain       = os.path.join(rootdir,'TRAIN')
rootdirtest        = os.path.join(rootdir,'TEST')

print('featuresSameLength = %s' % featuresSameLength)
print('phonemesSameLength = %s' % phonemesSameLength)
print('rootdir            = %s' % rootdir)
print('savedir            = %s' % savedir)
print('validIDs           = %s' % options.valid)

#------------------- Load Files --------------------
print('loading files')
def getFiles(rootdir,printFiles=False):
    files = {}
    for root, dirnames, filenames in os.walk(rootdir):
        for f in filenames:
            myname = '/'.join(root.split('/')[-2:] + [f])
            myid,filetype = myname.split('.')
            if filetype in ['PHN','TXT','WAV','WRD']:
                dr,spkr,st = myid.split('/')
                if not myid in list(files.keys()):
                    files[myid] = {}
                    files[myid]['dr'] = dr
                    files[myid]['spkr'] = spkr
                    files[myid]['sent'] = st
                    files[myid]['root'] = rootdir
                files[myid][filetype] = myname
                if printFiles:
                    print(myname)
    return files


alltrainfiles = getFiles(rootdirtrain)
testfiles = getFiles(rootdirtest)

#------------------- Load Validation SpeakerIDs --------------------
print('load validation speakerIDs')
vaids = np.loadtxt(options.valid,dtype=str)

trainfiles = {}
validfiles = {}
for k,v in alltrainfiles.items():
    if v['spkr'] in vaids:
        validfiles[k] = v
    else:
        trainfiles[k] = v

### check there are only 50 spkrs in validfiles, 
### and check that there is no overlap between trainfiles and validfiles
vaspkrs = set([v['spkr'] for k,v in validfiles.items()])
print("num valid speakers = %s" % len(vaspkrs))

trspkrs = set([v['spkr'] for k,v in trainfiles.items()])
print('len(vaspkrs.intersection(trspkrs)) = %s' % len(vaspkrs.intersection(trspkrs)))

#------------------- Process Phonemes & Words --------------------
print('parse phonemes & words')
def parseFile(x,filetype):
    with open(os.path.join(x['root'],x[filetype])) as f:
        lines = f.read().split('\n')
    return [l.split()[-1] for l in lines if len(l) > 0]

def parseAllFiles(files,filetype,keyname):
    for k,f in files.items():
        f[keyname] = parseFile(f,filetype)
        
def addEOStag(files):
    eos = '<EOS>'
    for k,f in files.items():
        f['phonemes'].append(eos) 

def makeLabelsSameLength(train,valid,test,eos='<EOS>'):
    maxlength = 0
    for k,f in train.items():
        maxlength = max(maxlength,len(f['phonemes']))
    
    for k,f in valid.items():
        assert len(f['phonemes']) <= maxlength, 'uh-oh'
    for k,f in test.items():
        assert len(f['phonemes']) <= maxlength, 'uh-oh'
        
    maxlength = maxlength
    for k,f in train.items():
        mylen = len(f['phonemes'])
        f['phonemes'] = f['phonemes'] + [eos]*(maxlength-mylen)  
        f['label_flag'] = np.zeros(maxlength)  
        f['label_flag'][:mylen] = 1
    for k,f in valid.items():
        mylen = len(f['phonemes'])
        f['phonemes'] = f['phonemes'] + [eos]*(maxlength-mylen) 
        f['label_flag'] = np.zeros(maxlength)  
        f['label_flag'][:mylen] = 1   
    for k,f in test.items():
        mylen = len(f['phonemes'])
        f['phonemes'] = f['phonemes'] + [eos]*(maxlength-mylen)
        f['label_flag'] = np.zeros(maxlength)  
        f['label_flag'][:mylen] = 1
        
parseAllFiles(trainfiles,'PHN','phonemes')
parseAllFiles(validfiles,'PHN','phonemes')
parseAllFiles(testfiles,'PHN','phonemes')

parseAllFiles(trainfiles,'WRD','words')
parseAllFiles(validfiles,'WRD','words')
parseAllFiles(testfiles,'WRD','words')

addEOStag(trainfiles)
addEOStag(validfiles)
addEOStag(testfiles)

if phonemesSameLength:
    makeLabelsSameLength(trainfiles,validfiles,testfiles)
      
#------------------- generate phoneme vocab --------------------
print('generate phoneme vocab')
phonemesTrain = set()
for k,f in trainfiles.items():
    for p in f['phonemes']:
        phonemesTrain.add(p)
for k,f in validfiles.items():
    for p in f['phonemes']:
        phonemesTrain.add(p)

phonemesTest = set()
for k,f in testfiles.items():
    for p in f['phonemes']:
        phonemesTest.add(p)

phonemes = {p:i+1 for i,p in enumerate(phonemesTrain)}

#------------------- digitize phonemes --------------------
print('digitize phonemes')
def digitizePhonemes(files):
    lengths = set()
    for k,f in files.items():
        f['phonemeLabels'] = [phonemes[p] for p in f['phonemes']]
        lengths.add(len(f['phonemeLabels']))

digitizePhonemes(trainfiles)
digitizePhonemes(validfiles)
digitizePhonemes(testfiles)

#------------------- generate features --------------------
def logmel(filename,n_fft=2048,hop_length=512):
    data, fs = librosa.load(filename)
    melspectrogram = librosa.feature.melspectrogram(y=data, sr=fs, n_fft=n_fft, hop_length=hop_length)
    logmel = librosa.core.logamplitude(melspectrogram)[:40,:]
    delta1 = librosa.feature.delta(logmel[:40,:],order=1)
    delta2 = librosa.feature.delta(logmel[:40,:],order=2)
    energy = librosa.feature.rmse(y=data)
    features = np.vstack((logmel,delta1,delta2,energy))
    return features.T

def CQT(filename, fmin=None, n_bins=84, hop_length=512):
    data, fs = librosa.load(filename)
    cqt = librosa.cqt(data, sr=fs, fmin=fmin, n_bins=n_bins, hop_length=hop_length)
    delta1 = librosa.feature.delta(cqt[24:,:],order=1)
    delta2 = librosa.feature.delta(cqt[24:,:],order=2)
    energy = librosa.feature.rmse(y=data)
    features = np.vstack((cqt,delta1,delta2,energy))
    return features.T

def getFeatures(files,func=logmel,**kwargs):
    for k,f in files.items():
        filename = os.path.join(f['root'],f['WAV'])
        f['features'] = func(filename,**kwargs)
        
def normalizeFeatures(train,valid,test,pad=10,use_samelength=False):
    maxlength = 0
    featurelist = []
    for k,f in train.items():
        maxlength = max(maxlength,len(f['features']))
        featurelist.append(f['features'])
    featurelist = np.vstack(featurelist)
    mean = featurelist.mean(axis=0)
    std = featurelist.std(axis=0)
    
    def normalize_and_pad(files):
        for k,f in files.items():
            mylen = len(f['features'])
            padding = np.zeros((pad,f['features'].shape[1]))
            f['features'] = (f['features']-mean)/std
            f['features'] = np.vstack([padding,f['features'],padding])
            if use_samelength:
                if mylen < maxlength:
                    extra_padding = np.zeros((maxlength-mylen,f['features'].shape[1]))
                    f['features'] = np.vstack([f['features'],extra_padding])
    
    normalize_and_pad(train)
    normalize_and_pad(valid)
    normalize_and_pad(test)
    
    return mean, std 

#------------------- pickle --------------------
def pickleIt(X,outputName):
    with open(outputName,'wb') as f:
        pickle.dump(X,f)

#------------------- To HDF5 --------------------
def toHDF5(allfiles,filename):
    with h5py.File(filename,'w') as h:
        for g,files in allfiles.items():
            grp = h.create_group(g)
            template = files[list(files.keys())[0]]
            sequenceLength,featureDepth = template['features'].shape
            
            if featuresSameLength and phonemesSameLength:
                features = [f['features'].reshape(1,sequenceLength,featureDepth) for f in list(files.values())]
                labels = [np.array(f['phonemeLabels']).reshape(1,-1) for f in list(files.values())]
                label_flags = [np.array(f['label_flag']).reshape(1,-1) for f in list(files.values())]
                grp['x'] = np.vstack(features)
                grp['y'] = np.vstack(labels)
                grp['ymask'] = np.vstack(label_flags)
            else:
                for k,f in enumerate(files.values()):
                    mygrp      = grp.create_group(str(k))
                    mygrp['x'] = f['features']
                    mygrp['y'] = np.array(f['phonemeLabels'])

#------------------- logmel features --------------------
print('generating logmel features')
getFeatures(trainfiles,func=logmel)
getFeatures(validfiles,func=logmel)
getFeatures(testfiles,func=logmel)
logmel_mean, logmel_std = normalizeFeatures(trainfiles,validfiles,testfiles,pad=4,use_samelength=featuresSameLength)

allfiles = {
    'train' : trainfiles,
    'valid' : validfiles,
    'test'  : testfiles
}

toHDF5(allfiles,os.path.join(savedir,'logmel.h5'))
pickleIt([logmel_mean, logmel_std],os.path.join(savedir,'logmel_mean_std.pkl'))

#------------------- CQT features --------------------
print('generating cqt features')
getFeatures(trainfiles,func=CQT)
getFeatures(validfiles,func=CQT)
getFeatures(testfiles,func=CQT)
cqt_mean, cqt_std = normalizeFeatures(trainfiles,validfiles,testfiles,pad=4,use_samelength=featuresSameLength)

allfiles = {
    'train' : trainfiles,
    'valid' : validfiles,
    'test'  : testfiles
}

toHDF5(allfiles,os.path.join(savedir,'cqt.h5'))
pickleIt([cqt_mean, cqt_std],os.path.join(savedir,'cqt_mean_std.pkl'))
