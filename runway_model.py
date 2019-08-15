import os
import torch
from PIL import Image
from collections import OrderedDict
import numpy as np
import pathlib
from glob import glob
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from data.base_dataset import get_params, get_transform
import runway

opt = TestOptions().parse(save=False)
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
opt.resize_or_crop = 'none'
opt.use_features = False
opt.no_instance = True
opt.label_nc = 0

@runway.setup(options={'checkpoints_root': runway.file(is_directory=True)})
def setup(opts):
    opt.name = opts['checkpoints_root'].split('/')[-1]
    opt.checkpoints_dir = os.path.join(opts['checkpoints_root'], '..')
    model = create_model(opt)
    return model

@runway.command('generate', inputs={'image': runway.image}, outputs={'image': runway.image})
def generate(model, inputs):
    label = inputs['image']
    params = get_params(opt, label.size)
    transform_label = get_transform(
        opt, params, method=Image.NEAREST, normalize=False
    )
    label_tensor = transform_label(label)
    label_tensor = label_tensor.unsqueeze(0)
    generated = model.inference(label_tensor, None)
    torch.cuda.synchronize()
    im = util.tensor2im(generated.data[0])
    return Image.fromarray(im)


if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8888, model_options={'checkpoints_root': './checkpoints/shining'})
