# monkey patch ultralytics to match the fork in the original source
import cv2
from tqdm import tqdm
import numpy as np
import json
import torch
import ultralytics
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.modules.head import Detect, Segment
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.yolo.utils import DEFAULT_CFG, LOGGER, RANK, SETTINGS, TQDM_BAR_FORMAT, callbacks, colorstr, emojis
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.ops import Profile
from ultralytics.yolo.utils.torch_utils import de_parallel, select_device, smart_inference_mode
from ultralytics.yolo.utils.tal import dist2bbox, make_anchors

from ultralytics.nn.modules.block import DFL, Proto
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from ultralytics.nn.modules.utils import bias_init_with_prob, linear_init_

def scale_image(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size

    Args:
      masks (torch.Tensor): resized and padded masks/images, [h, w, num]/[h, w, 3].
      im0_shape (tuple): the original image shape
      ratio_pad (tuple): the ratio of the padding to the original image.

    Returns:
      masks (torch.Tensor): The masks that are being returned.
    """
    print("warning: scale_image was patched by mobile_sam_v2")
    # Rescale coordinates (xyxy) from im1_shape to im0_shape
    im1_shape = masks.shape
    if im1_shape[:2] == im0_shape[:2]:
        return masks
    if ratio_pad is None:  # calculate from im0_shape
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    top, left = int(pad[1]), int(pad[0])  # y, x
    bottom, right = int(im1_shape[0] - pad[1]), int(im1_shape[1] - pad[0])

    if len(masks.shape) < 2:
        raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
    masks = masks[top:bottom, left:right]
    # masks = masks.permute(2, 0, 1).contiguous()
    # masks = F.interpolate(masks[None], im0_shape[:2], mode='bilinear', align_corners=False)[0]
    # masks = masks.permute(1, 2, 0).contiguous()
    cv_limit = 512
    if masks.shape[2] <= cv_limit:
        masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]))
    else:
        # split masks array on batches with max size 512 along channel axis, resize and merge them back
        masks = np.concatenate([cv2.resize(masks[:, :, i:min(i + cv_limit, masks.shape[2])], (im0_shape[1], im0_shape[0]))
                                for i in range(0, masks.shape[2], cv_limit)], axis=2)
    if len(masks.shape) == 2:
        masks = masks[:, :, None]

    return masks

@smart_inference_mode()
def BaseValidator__call__(self, trainer=None, model=None):
    """
    Supports validation of a pre-trained model if passed or a model being trained
    if trainer is passed (trainer gets priority).
    """
    print("warning: BaseValidator.__call__ was patched by mobile_sam_v2")
    self.training = trainer is not None
    if self.training:
        self.device = trainer.device
        self.data = trainer.data
        model = trainer.ema.ema or trainer.model
        self.args.half = self.device.type != 'cpu'  # force FP16 val during training
        model = model.half() if self.args.half else model.float()
        self.model = model
        self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
        self.args.plots = trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
        model.eval()
    else:
        callbacks.add_integration_callbacks(self)
        self.run_callbacks('on_val_start')
        assert model is not None, 'Either trainer or model is needed for validation'
        self.device = select_device(self.args.device, self.args.batch)
        self.args.half &= self.device.type != 'cpu'
        model = AutoBackend(model, device=self.device, dnn=self.args.dnn, data=self.args.data, fp16=self.args.half)
        self.model = model
        stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
        imgsz = check_imgsz(self.args.imgsz, stride=stride)
        if engine:
            self.args.batch = model.batch_size
        else:
            self.device = model.device
            if not pt and not jit:
                self.args.batch = 1  # export.py models default to batch-size 1
                LOGGER.info(f'Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models')

        if isinstance(self.args.data, str) and self.args.data.endswith('.yaml'):
            self.data = check_det_dataset(self.args.data)
        elif self.args.task == 'classify':
            self.data = check_cls_dataset(self.args.data, split=self.args.split)
        else:
            raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

        if self.device.type == 'cpu':
            self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
        if not pt:
            self.args.rect = False
        self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

        model.eval()
        model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

    dt = Profile(), Profile(), Profile(), Profile()
    n_batches = len(self.dataloader)
    desc = self.get_desc()
    # NOTE: keeping `not self.training` in tqdm will eliminate pbar after segmentation evaluation during training,
    # which may affect classification task since this arg is in yolov5/classify/val.py.
    # bar = tqdm(self.dataloader, desc, n_batches, not self.training, bar_format=TQDM_BAR_FORMAT)
    bar = tqdm(self.dataloader, desc, n_batches, bar_format=TQDM_BAR_FORMAT)
    self.init_metrics(de_parallel(model))
    self.jdict = []  # empty before each val
    for batch_i, batch in enumerate(bar):
        self.run_callbacks('on_val_batch_start')
        self.batch_i = batch_i
        # Preprocess
        with dt[0]:
            batch = self.preprocess(batch)

        # Inference
        with dt[1]:
            preds = model(batch['img'])

        # Loss
        with dt[2]:
            if self.training:
                self.loss += trainer.criterion(preds, batch)[1]

        # Postprocess
        with dt[3]:
            preds = self.postprocess(preds)
        # import pdb;pdb.set_trace()
        try:
            self.update_metrics(preds, batch)
        except:
            with open('wrong_file_1.txt', 'a') as f:
                f.write(str(batch['im_file']))
                f.write('\n')
            # continue
        if self.args.plots and batch_i < 3:
            self.plot_val_samples(batch, batch_i)
            self.plot_predictions(batch, preds, batch_i)
        # print(self.args.save_json, self.jdict)
        self.run_callbacks('on_val_batch_end')
        # if self.args.save_json and self.jdict:
        #     with open(str(self.save_dir / 'tmp_predictions.json'), 'w') as f:
        #         # LOGGER.info(f'Saving {f.name}...')
        #         json.dump(self.jdict, f)  # flatten and save
    stats = self.get_stats()
    self.check_stats(stats)
    self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1E3 for x in dt)))
    self.finalize_metrics()
    self.print_results()
    self.run_callbacks('on_val_end')
    if self.training:
        model.float()
        results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix='val')}
        return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
    else:
        LOGGER.info('Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image' %
                    tuple(self.speed.values()))

        if self.args.save_json and self.jdict:
            with open(str(self.save_dir / 'predictions.json'), 'w') as f:
                LOGGER.info(f'Saving {f.name}...')
                json.dump(self.jdict, f)  # flatten and save
            stats = self.eval_json(stats)  # update stats
        if self.args.plots or self.args.save_json:
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        return stats

def Detect__init__(self, nc=80, ch=()):  # detection layer
    print("warning: Detect.__init__ was patched by mobile_sam_v2")
    super(Detect, self).__init__()
    self.nc = nc  # number of classes
    self.nl = len(ch)  # number of detection layers
    self.reg_max = 26  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
    self.no = nc + self.reg_max * 4  # number of outputs per anchor
    self.stride = torch.zeros(self.nl)  # strides computed during build
    c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
    self.cv2 = torch.nn.ModuleList(
        torch.nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), torch.nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch)
    self.cv3 = torch.nn.ModuleList(torch.nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), torch.nn.Conv2d(c3, self.nc, 1)) for x in ch)
    self.dfl = DFL(self.reg_max) if self.reg_max > 1 else torch.nn.Identity()

segment_original_init = Segment.__init__
def Segment__init__(self, nc=80, nm=32, npr=256, ch=()):
    print("warning: Segment.__init__ was patched by mobile_sam_v2")
    segment_original_init(self, nc, nm, npr, ch)
    # remove protos (to match the comment)
    self.protos = None

def Segment_forward(self, x):
    """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
    print("warning: Segment.forward was patched by mobile_sam_v2")
    #p = self.proto(x[0])  # mask protos #mobilesamv2 change
    p=0
    # import pdb;pdb.set_trace()
    bs = x[0].shape[0]  # batch size

    mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
    x = self.detect(self, x)
    if self.training:
        return x, mc, p
    return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))

# workaround for the weights_only issue
def torch_safe_load(weight):
    """
    This function attempts to load a PyTorch model with the torch.load() function. If a ModuleNotFoundError is raised,
    it catches the error, logs a warning message, and attempts to install the missing module via the
    check_requirements() function. After installation, the function again attempts to load the model using torch.load().

    Args:
        weight (str): The file path of the PyTorch model.

    Returns:
        (dict): The loaded PyTorch model.
    """
    print("warning: torch_safe_load was patched by mobile_sam_v2")
    import ultralytics.yolo.utils.downloads
    from ultralytics.nn.tasks import check_suffix, check_requirements
    from ultralytics.yolo.utils.downloads import attempt_download_asset

    check_suffix(file=weight, suffix='.pt')
    file = attempt_download_asset(weight)  # search online if missing locally
    try:
        return torch.load(file, map_location='cpu', weights_only=False), file  # load
    except ModuleNotFoundError as e:  # e.name is missing module name
        if e.name == 'models':
            raise TypeError(
                emojis(f'ERROR ❌️ {weight} appears to be an Ultralytics YOLOv5 model originally trained '
                       f'with https://github.com/ultralytics/yolov5.\nThis model is NOT forwards compatible with '
                       f'YOLOv8 at https://github.com/ultralytics/ultralytics.'
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")) from e
        LOGGER.warning(f"WARNING ⚠️ {weight} appears to require '{e.name}', which is not in ultralytics requirements."
                       f"\nAutoInstall will run now for '{e.name}' but this feature will be removed in the future."
                       f"\nRecommend fixes are to train a new model using the latest 'ultralytics' package or to "
                       f"run a command with an official YOLOv8 model, i.e. 'yolo predict model=yolov8n.pt'")
        check_requirements(e.name)  # install missing module

        return torch.load(file, map_location='cpu'), file  # load


def patch_ultralytics():
    import ultralytics.yolo.utils.ops as yolo_ops
    import ultralytics.nn.tasks as nn_tasks
    from ultralytics.yolo.engine.validator import BaseValidator

    yolo_ops.scale_image = scale_image
    BaseValidator.__call__ = BaseValidator__call__
    Detect.__init__ = Detect__init__
    Segment.__init__ = Segment__init__
    Segment.forward = Segment_forward

    nn_tasks.torch_safe_load = torch_safe_load
