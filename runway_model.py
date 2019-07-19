import os
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
opt.netG = 'local'
opt.ngf = 32
opt.resize_or_crop = 'none'

@runway.setup(options={'checkpoints_root': runway.file(is_directory=True)})
def setup(opts):
    opt.name = opts['checkpoints_root'].split('/')[-1]
    opt.checkpoints_dir = os.path.join(opts['checkpoints_root'], '..')
    model = create_model(opt)
    return model

label_to_id = {
    'unlabeled': 0,
    'ground': 6, 
    'road': 7,
    'sidewalk': 8,
    'parking': 9,
    'rail track': 10,
    'parking': 9,
    'rail track': 10,
    'building': 11,
    'wall': 12,
    'fence': 13,
    'guard rail': 14,
    'bridge': 15,
    'tunnel': 16,
    'pole': 17,
    'traffic light': 19,
    'traffic sign': 20,
    'vegetation': 21,
    'terrain': 22,
    'sky': 23,
    'person': 24,
    'rider': 25,
    'car': 26,
    'truck': 27,
    'bus': 28,
    'caravan': 29,
    'trailer': 30,
    'train': 31,
    'motorcycle': 32,
    'bicycle': 33
}

@runway.command('generate', inputs={'segmentation': runway.segmentation(label_to_id=label_to_id)}, outputs={'image': runway.image})
def generate(model, inputs):
    label = inputs['segmentation']
    params = get_params(opt, label.size)
    transform_label = get_transform(
        opt, params, method=Image.NEAREST, normalize=False
    )
    label_tensor = transform_label(label) * 255.0
    inst_tensor = transform_label(label)
    label_tensor = label_tensor.unsqueeze(0)
    inst_tensor = inst_tensor.unsqueeze(0)
    generated = model.inference(label_tensor, inst_tensor)
    im = util.tensor2im(generated.data[0])
    return im


if __name__ == '__main__':
    runway.run(host='0.0.0.0', port=8888, model_options={'checkpoints_root': './checkpoint'})
