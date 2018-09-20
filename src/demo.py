import sys
import torch
from opts import opts
import ref
from utils.debugger import Debugger
from utils.eval import getPreds
import cv2
import numpy as np
from timeit import default_timer as timer
import torch.onnx

def main():
  opt = opts().parse()
  model = load_model()
  img = cv2.imread(opt.demo)
  with_img(img, model, debug = True)


def load_model(path = 'hgreg-3d.pth'):
  # opt = opts().parse()
  # if opt.loadModel != 'none':
    # model = torch.load(opt.loadModel).cuda()
  # else:
  model = torch.load(path).cuda()
  return model
  

def with_img(img, model, debug = False):
  input = torch.from_numpy(img.transpose(2, 0, 1)).float() / 256.
  

  start = timer()
  input = input.view(1, input.size(0), input.size(1), input.size(2))
  input_var = torch.autograd.Variable(input).float().cuda()
  # torch.onnx.export(model, input_var, "/tmp/model.onnx", verbose=True)
  end = timer()
  # print('copying to gpu: ', end - start)
  start = timer()
  output = model(input_var)
  end = timer()
  # print('running model: ', end - start)

  start = timer()
  pred = getPreds((output[-2].data).cpu().numpy())[0] * 4
  end = timer()
  # print('pred: ',pred)
  # print(end - start)
  reg = (output[-1].data).cpu().numpy().reshape(pred.shape[0], 1)
  if debug:
    print('reg: ', reg)
    debugger = Debugger()
    debugger.addImg((input[0].numpy().transpose(1, 2, 0)*256).astype(np.uint8))
    debugger.addPoint2D(pred, (255, 0, 0))
    debugger.addPoint3D(np.concatenate([pred, (reg + 1) / 2. * 256], axis = 1))
    debugger.showImg(pause = True)
    debugger.show3D()
  return pred, img

if __name__ == '__main__':
  main()
