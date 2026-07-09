# Copyright (c) Opendatalab. All rights reserved.
import sys

import numpy as np
import time
import torch
import torch_npu

from ...pytorchocr.base_ocr_v20 import BaseOCRV20
from . import pytorchocr_utility as utility
from ...pytorchocr.data import create_operators, transform
from ...pytorchocr.postprocess import build_post_process
import torch.nn.functional as F


class TextDetector(BaseOCRV20):
    def __init__(self, args, **kwargs):
        self.args = args
        self.det_algorithm = args.det_algorithm
        self.device = args.device

        pre_process_list = [{
            'DetResizeForTest': {
                'limit_side_len': args.det_limit_side_len,
                'limit_type': args.det_limit_type,
                'max_side_limit': args.det_max_side_limit,
            }
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        if self.det_algorithm == "DB":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = args.use_dilation
            postprocess_params["score_mode"] = args.det_db_score_mode
            postprocess_params["box_type"] = args.det_box_type
        elif self.det_algorithm == "DB++":
            postprocess_params['name'] = 'DBPostProcess'
            postprocess_params["thresh"] = args.det_db_thresh
            postprocess_params["box_thresh"] = args.det_db_box_thresh
            postprocess_params["max_candidates"] = 1000
            postprocess_params["unclip_ratio"] = args.det_db_unclip_ratio
            postprocess_params["use_dilation"] = args.use_dilation
            postprocess_params["score_mode"] = args.det_db_score_mode
            postprocess_params["box_type"] = args.det_box_type
            pre_process_list[1] = {
                'NormalizeImage': {
                    'std': [1.0, 1.0, 1.0],
                    'mean':
                        [0.48109378172549, 0.45752457890196, 0.40787054090196],
                    'scale': '1./255.',
                    'order': 'hwc'
                }
            }
        elif self.det_algorithm == "EAST":
            postprocess_params['name'] = 'EASTPostProcess'
            postprocess_params["score_thresh"] = args.det_east_score_thresh
            postprocess_params["cover_thresh"] = args.det_east_cover_thresh
            postprocess_params["nms_thresh"] = args.det_east_nms_thresh
        elif self.det_algorithm == "SAST":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'resize_long': args.det_limit_side_len
                }
            }
            postprocess_params['name'] = 'SASTPostProcess'
            postprocess_params["score_thresh"] = args.det_sast_score_thresh
            postprocess_params["nms_thresh"] = args.det_sast_nms_thresh
            self.det_sast_polygon = args.det_sast_polygon
            if self.det_sast_polygon:
                postprocess_params["sample_pts_num"] = 6
                postprocess_params["expand_scale"] = 1.2
                postprocess_params["shrink_ratio_of_width"] = 0.2
            else:
                postprocess_params["sample_pts_num"] = 2
                postprocess_params["expand_scale"] = 1.0
                postprocess_params["shrink_ratio_of_width"] = 0.3
        elif self.det_algorithm == "PSE":
            postprocess_params['name'] = 'PSEPostProcess'
            postprocess_params["thresh"] = args.det_pse_thresh
            postprocess_params["box_thresh"] = args.det_pse_box_thresh
            postprocess_params["min_area"] = args.det_pse_min_area
            postprocess_params["box_type"] = args.det_pse_box_type
            postprocess_params["scale"] = args.det_pse_scale
            self.det_pse_box_type = args.det_pse_box_type
        elif self.det_algorithm == "FCE":
            pre_process_list[0] = {
                'DetResizeForTest': {
                    'rescale_img': [1080, 736]
                }
            }
            postprocess_params['name'] = 'FCEPostProcess'
            postprocess_params["scales"] = args.scales
            postprocess_params["alpha"] = args.alpha
            postprocess_params["beta"] = args.beta
            postprocess_params["fourier_degree"] = args.fourier_degree
            postprocess_params["box_type"] = args.det_fce_box_type
        else:
            print("unknown det_algorithm:{}".format(self.det_algorithm))
            sys.exit(0)

        # Set normalization parameters according to det_algorithm
        self.shape_311 = (3, 1, 1)
        if self.det_algorithm == "DB++":
            norm_mean = np.array([0.48109378172549, 0.45752457890196, 0.40787054090196]).reshape(self.shape_311).astype('float32')
            norm_std = np.array([1.0, 1.0, 1.0]).reshape(self.shape_311).astype('float32')
        else:
            norm_mean = np.array([0.485, 0.456, 0.406]).reshape(self.shape_311).astype('float32')
            norm_std = np.array([0.229, 0.224, 0.225]).reshape(self.shape_311).astype('float32')

        self.scale = torch.from_numpy(np.float32(1.0 / 255.0) / norm_std).to(self.device)
        self.mean = torch.from_numpy(norm_mean / norm_std).to(self.device)

        # SAST uses resize_long; other algorithms (DB/DB++/EAST/PSE/FCE) use limit_side_len + limit_type.
        # Note: although FCE's pre_process_list has 'rescale_img', DetResizeForTest does not recognize that key,
        # so it actually goes to the else branch → limit_side_len=736, limit_type='min'.
        if self.det_algorithm == "SAST":
            self.resize_type = 2
            self.resize_long = args.det_limit_side_len
            self.max_side_limit = args.det_max_side_limit
        else:
            self.resize_type = 0
            self.limit_side_len = args.det_limit_side_len
            self.limit_type = args.det_limit_type
            self.max_side_limit = args.det_max_side_limit

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)

        self.weights_path = args.det_model_path
        self.yaml_path = args.det_yaml_path
        network_config = utility.get_arch_config(self.weights_path)
        super(TextDetector, self).__init__(network_config, **kwargs)
        self.load_pytorch_weights(self.weights_path)
        self.net.eval()
        if self.device != 'cpu':
            self.net.to(self.device)

        for module in self.net.modules():
            if hasattr(module, 'rep'):
                module.rep()

    def _should_only_clip_det_res(self):
        if self.det_algorithm == "SAST" and getattr(self, "det_sast_polygon", False):
            return True
        if self.det_algorithm in ["DB", "DB++", "PSE", "FCE"]:
            return getattr(self.postprocess_op, "box_type", "quad") == "poly"
        return False

    def _filter_det_res(self, dt_boxes, image_shape):
        if self._should_only_clip_det_res():
            return self.filter_tag_det_res_only_clip(dt_boxes, image_shape)
        return self.filter_tag_det_res(dt_boxes, image_shape)

    def _compute_resize_params(self, h, w):
        """
        Replicates the resize logic of DetResizeForTest, supporting two resize_type:
          type0: limit_side_len + limit_type (max/min/resize_long) + max_side_limit, aligned to multiples of 32
          type2: resize_long (SAST), aligned to multiples of 128

        Returns:
            (resize_h, resize_w, ratio_h, ratio_w) or (None, None, None, None)
        """
        if self.resize_type == 2:
            # SAST: resize_long, aligned to multiples of 128
            resize_h, resize_w = h, w
            if resize_h > resize_w:
                ratio = float(self.resize_long) / resize_h
            else:
                ratio = float(self.resize_long) / resize_w
            resize_h = int(resize_h * ratio)
            resize_w = int(resize_w * ratio)
            max_stride = 128
            resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
            resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
            ratio_h = resize_h / float(h)
            ratio_w = resize_w / float(w)
            return resize_h, resize_w, ratio_h, ratio_w

        # type0: limit_side_len + limit_type + max_side_limit, aligned to multiples of 32
        limit_side_len = self.limit_side_len
        if self.limit_type == 'max':
            if max(h, w) > limit_side_len:
                if h > w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'min':
            if min(h, w) < limit_side_len:
                if h < w:
                    ratio = float(limit_side_len) / h
                else:
                    ratio = float(limit_side_len) / w
            else:
                ratio = 1.
        elif self.limit_type == 'resize_long':
            ratio = float(limit_side_len) / max(h, w)
        else:
            raise Exception('not support limit type: {}'.format(self.limit_type))

        resize_h = int(h * ratio)
        resize_w = int(w * ratio)

        if max(resize_h, resize_w) > self.max_side_limit:
            ratio = float(self.max_side_limit) / max(resize_h, resize_w)
            resize_h = int(resize_h * ratio)
            resize_w = int(resize_w * ratio)

        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)

        if int(resize_w) <= 0 or int(resize_h) <= 0:
            return None, None, None, None

        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return resize_h, resize_w, ratio_h, ratio_w

    def _batch_process_same_size(self, img_list):
        """
        Batch processes images of the same size.
        All images in img_list must have the same shape, otherwise an assertion failure is raised.

        Args:
            img_list: list of images of the same shape (each shape = [H, W, C], uint8 numpy)

        Returns:
            batch_results: list of batch processing results, each element is (dt_boxes, elapse)
            total_elapse: total elapsed time
        """
        t0 = time.perf_counter()
        if not img_list:
            return [], 0

        # Assertion: all images must have the same shape
        ref_shape = img_list[0].shape
        assert all(x.shape == ref_shape for x in img_list), \
            f"_batch_process_same_size: images have different shapes, " \
            f"expected {ref_shape}, got {[x.shape for x in img_list]}"

        starttime = time.time()
        h, w, c = ref_shape

        # Compute resize parameters
        resize_h, resize_w, ratio_h, ratio_w = self._compute_resize_params(h, w)
        if resize_h is None:
            return [(None, 0)] * len(img_list), 0

        batch_shapes = [np.array([h, w, ratio_h, ratio_w])] * len(img_list)

        # Build batch tensor: np.stack → BHWC → BCHW → NPU
        batch_numpy = np.stack(img_list, axis=0)  # [N, H, W, C]
        tensor_bchw = torch.from_numpy(batch_numpy).permute(0, 3, 1, 2).to(self.device, torch.float16, non_blocking=True)

        need_resize = not (tensor_bchw.shape[-1] == int(resize_w) and tensor_bchw.shape[-2] == int(resize_h))

        with torch.inference_mode():
            if need_resize:
                # Process in chunks with F.interpolate to reduce peak GPU memory
                chunk_size = 64
                resized_list = []
                for i in range(0, len(img_list), chunk_size):
                    chunk_tensor = tensor_bchw[i: i + chunk_size]
                    resized_chunk = F.interpolate(
                        chunk_tensor,
                        size=(int(resize_h), int(resize_w)),
                        mode='bilinear',
                        align_corners=False
                    )
                    resized_list.append(resized_chunk)
                final_tensor = torch.cat(resized_list, dim=0)
            else:
                final_tensor = tensor_bchw

            # scale = (1/255) / std, mean = mean_ / std
            final_tensor = final_tensor * self.scale - self.mean

            t1 = time.perf_counter()
            # print(f"[Kenny Logs] Preprocess spend time: {(t1 - t0)*1000}/ms")
            outputs = self.net(final_tensor)
            t2 = time.perf_counter()
            # print(f"[Kenny Logs] Inference spend time: {(t2 - t1)*1000}/ms")


        preds = {}
        if self.det_algorithm == "EAST":
            preds['f_geo'] = outputs['f_geo'].cpu().numpy()
            preds['f_score'] = outputs['f_score'].cpu().numpy()
        elif self.det_algorithm == 'SAST':
            preds['f_border'] = outputs['f_border'].cpu().numpy()
            preds['f_score'] = outputs['f_score'].cpu().numpy()
            preds['f_tco'] = outputs['f_tco'].cpu().numpy()
            preds['f_tvo'] = outputs['f_tvo'].cpu().numpy()
        elif self.det_algorithm in ['DB', 'PSE', 'DB++']:
            # Pass tensor directly to postprocess so that DBPostProcess can do threshold comparisons on NPU,
            # avoiding the overhead of full pred.cpu().numpy() copy.
            preds['maps'] = outputs['maps']
        elif self.det_algorithm == 'FCE':
            for i, (k, output) in enumerate(outputs.items()):
                preds['level_{}'.format(i)] = output.cpu().numpy()
        else:
            raise NotImplementedError

        # Pass the entire batch to postprocess_op at once to avoid repeated threshold computations per image
        batch_shapes_np = np.stack(batch_shapes, axis=0)
        post_results = self.postprocess_op(preds, batch_shapes_np)

        # Filter per image
        batch_results = []
        total_elapse = time.time() - starttime

        for i in range(len(img_list)):
            dt_boxes = post_results[i]['points']
            dt_boxes = self._filter_det_res(dt_boxes, img_list[i].shape)
            batch_results.append((dt_boxes, total_elapse / len(img_list)))
        t3 = time.perf_counter()
        # print(f"[Kenny Logs] Postprocess spend time: {(t3 - t2)*1000}/ms")

        return batch_results, total_elapse

    def batch_predict(self, img_list, max_batch_size=8):
        """
        Batch prediction method supporting detection on multiple images simultaneously

        Args:
            img_list: list of images
            max_batch_size: maximum batch size

        Returns:
            batch_results: list of batch results, each element is (dt_boxes, elapse)
        """
        if not img_list:
            return []

        batch_results = []

        # Group by shape, process within each group
        from collections import OrderedDict
        shape_groups = OrderedDict()
        for idx, img in enumerate(img_list):
            key = img.shape
            if key not in shape_groups:
                shape_groups[key] = []
            shape_groups[key].append(img)

        for shape, imgs in shape_groups.items():
            # Further split images of the same shape by max_batch_size
            # print(f"[Kenny Logs] shape: {shape} imgs={len(imgs)}")
            for i in range(0, len(imgs), max_batch_size):
                batch_imgs = imgs[i:i + max_batch_size]
                batch_dt_boxes, batch_elapse = self._batch_process_same_size(batch_imgs)
                batch_results.extend(batch_dt_boxes)

        return batch_results

    def order_points_clockwise(self, pts):
        """
        reference from: https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py
        # sort the points based on their x-coordinates
        """
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
        (tr, br) = rightMost

        rect = np.array([tl, tr, br, bl], dtype="float32")
        return rect

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.order_points_clockwise(box)
            box = self.clip_det_res(box, img_height, img_width)
            rect_width = int(np.linalg.norm(box[0] - box[1]))
            rect_height = int(np.linalg.norm(box[0] - box[3]))
            if rect_width <= 3 or rect_height <= 3:
                continue
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = np.array(box)
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        # Polygon detectors may emit a variable number of points per box,
        # so this path must preserve a ragged outer container.
        return dt_boxes_new

    def __call__(self, img):
        ori_shape = img.shape
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        starttime = time.time()

        with torch.inference_mode():
            inp = torch.from_numpy(img)
            inp = inp.to(self.device)
            outputs = self.net(inp)

        preds = {}
        if self.det_algorithm == "EAST":
            preds['f_geo'] = outputs['f_geo'].cpu().numpy()
            preds['f_score'] = outputs['f_score'].cpu().numpy()
        elif self.det_algorithm == 'SAST':
            preds['f_border'] = outputs['f_border'].cpu().numpy()
            preds['f_score'] = outputs['f_score'].cpu().numpy()
            preds['f_tco'] = outputs['f_tco'].cpu().numpy()
            preds['f_tvo'] = outputs['f_tvo'].cpu().numpy()
        elif self.det_algorithm in ['DB', 'PSE', 'DB++']:
            # Pass tensor directly to postprocess so that DBPostProcess can do threshold comparisons on NPU
            preds['maps'] = outputs['maps']
        elif self.det_algorithm == 'FCE':
            for i, (k, output) in enumerate(outputs.items()):
                preds['level_{}'.format(i)] = output
        else:
            raise NotImplementedError

        post_result = self.postprocess_op(preds, shape_list)
        dt_boxes = post_result[0]['points']
        dt_boxes = self._filter_det_res(dt_boxes, ori_shape)

        elapse = time.time() - starttime
        return dt_boxes, elapse
