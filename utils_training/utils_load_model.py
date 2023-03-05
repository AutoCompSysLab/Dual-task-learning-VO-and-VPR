import math
pi=math.pi
import os
import torch
import shutil

def load_model(model, modelname):
    preTrainDict = torch.load(modelname)
    model_dict = model.state_dict()
    #import pdb; pdb.set_trace()
    if modelname.endswith('pkl'):
            preTrainDictTemp = {k.replace('module', 'vonet'):v for k,v in preTrainDict.items() if k not in model_dict}
    else:
        preTrainDictTemp = {k:v for k,v in preTrainDict['state_dict'].items() if k in model_dict}

    if( 0 == len(preTrainDictTemp) ):
        print("Does not find any module to load. Try DataParallel version.")
        for k, v in preTrainDict.items():
            kk = k[7:]
            if ( kk in model_dict ):
                preTrainDictTemp[kk] = v

    if ( 0 == len(preTrainDictTemp) ):
        raise Exception("Could not load model from %s." % (modelname), "load_model")

    model_dict.update(preTrainDictTemp)
    model.load_state_dict(model_dict)
    print('Model loaded...')
    return model