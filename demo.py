import torch
import utils
import dataset
from PIL import Image

import models.crnn as crnn


model_path = './data/crnn.pth'
img_path = './data/demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = crnn.CRNN(32, 1, 37, 256)
model = model.to(device)
print('loading pretrained model from %s' % model_path)
# Legacy checkpoints may require full unpickling on PyTorch >= 2.6.
state_dict = torch.load(model_path, map_location=device, weights_only=False)
model.load_state_dict(state_dict)

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
image = image.to(device)
image = image.view(1, *image.size())

model.eval()
with torch.no_grad():
    preds = model(image)

_, preds = preds.max(2)
preds = preds.transpose(1, 0).contiguous().view(-1)

preds_size = torch.IntTensor([preds.size(0)])
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
