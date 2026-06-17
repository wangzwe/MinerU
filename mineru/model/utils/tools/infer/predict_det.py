# Copyright (c) Opendatalab. All rights reserved.
import sys

import numpy as np
import time
import torch
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

        self.shape = (3, 1, 1) 
        self.mean_ = np.array([0.485, 0.456, 0.406]).reshape(self.shape).astype('float32')
        self.std = np.array([0.229, 0.224, 0.225]).reshape(self.shape).astype('float32')
        self.scale = torch.from_numpy(np.float32(1.0 / 255.0) / self.std).to(self.device)
        self.mean =  torch.from_numpy(self.mean_ / self.std).to(self.device)
        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)

        self.weights_path = args.det_model_path
        self.yaml_path = args.det_yaml_path
        network_config = utility.get_arch_config(self.weights_path)
        super(TextDetector, self).__init__(network_config, **kwargs)
        self.load_pytorch_weights(self.weights_path)
        self.net.eval()
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

    def _batch_process_same_size(self, img_list):
        """
            对相同尺寸的图像进行批处理

            Args:
                img_list: 相同尺寸的图像列表

            Returns:
                batch_results: 批处理结果列表
                total_elapse: 总耗时
            """
        starttime = time.time()
        
        # 预处理所有图像
        batch_data = []
        batch_shapes = []
        ori_imgs = []
        shape = img_list[0].shape
        t=time.time()
        if len(img_list) > 1 and   all(x.shape == shape for x in img_list):
            h, w, c = img_list[0].shape
            if max(h, w) > self.args.det_limit_side_len:
                if h > w:
                    ratio = float(self.args.det_limit_side_len) / h
                else:
                    ratio = float(self.args.det_limit_side_len) / w
            else:
                ratio = 1.           
            resize_h = int(h * ratio)
            resize_w = int(w * ratio)

            if max(resize_h, resize_w) > self.args.det_max_side_limit:
                ratio = float(self.args.det_max_side_limit) / max(resize_h, resize_w)
                resize_h = int(resize_h * ratio)
                resize_w = int(resize_w * ratio)

            resize_h = max(int(round(resize_h / 32) * 32), 32)
            resize_w = max(int(round(resize_w / 32) * 32), 32)
            ratio_h = resize_h / float(h)
            ratio_w = resize_w / float(w)
            shape = np.array([h, w, ratio_h, ratio_w])
            batch_shapes = batch_shapes = [shape] * len(img_list)
            try:
                if int(resize_w) <= 0 or int(resize_h) <= 0:
                    return None, (None, None)
                batch_numpy = np.stack(img_list, axis=0)
                # img = cv2.resize(img, (int(resize_w), int(resize_h)))
                tensor_bchw = torch.from_numpy(batch_numpy).permute(0, 3, 1, 2).float().to(self.device)
                print(tensor_bchw.shape)
                chunk_size = 32#超参数
                if tensor_bchw.shape[-1]==int(resize_h) and tensor_bchw.shape[-2]==int(resize_h):
                    final_tensor = tensor_bchw* self.scale - self.mean
                else:

                    resized_list = []

                    for i in range(0, len(img_list), chunk_size):
                        # 利用张量切片，获取一小批数据
                        chunk_tensor = tensor_bchw[i : i + chunk_size]
                        
                        # 在 NPU 上对这一小批进行 Resize
                        # 此时显存峰值大幅降低
                        resized_chunk = F.interpolate(
                            chunk_tensor, 
                            size=(int(resize_h), int(resize_w)), 
                            mode='bilinear', 
                            align_corners=False
                        )
                        resized_list.append(resized_chunk)
                
                    # 拼接回来，得到最终的 [128, 3, 224, 224]
                    final_tensor = torch.cat(resized_list, dim=0)* self.scale - self.mean

                print("final_tensor shape",final_tensor.shape)
                with torch.no_grad():
                    outputs = self.net(final_tensor) 
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
                    t3=time.time()
                    print("输出shape",outputs['maps'].shape)
                    preds['maps'] = outputs['maps'].cpu().numpy()
                    print("cpu_numpy",time.time()-t3) 
                elif self.det_algorithm == 'FCE':
                    for i, (k, output) in enumerate(outputs.items()):
                        preds['level_{}'.format(i)] = output.cpu().numpy()
                else:
                    raise NotImplementedError

                # 后处理每个图像的结果
                batch_results = []
                total_elapse = time.time() - starttime
                t4=time.time()

                for i in range(len(img_list)):
                    # 提取单个图像的预测结果
                    single_preds = {}
                    for key, value in preds.items():
                        if isinstance(value, np.ndarray):
                            single_preds[key] = value[i:i + 1]  # 保持批次维度
                        else:
                            single_preds[key] = value

                    # 后处理
                    post_result = self.postprocess_op(single_preds, batch_shapes[i:i + 1])
                    dt_boxes = post_result[0]['points']

                    # 过滤和裁剪检测框
                    dt_boxes = self._filter_det_res(dt_boxes, shape)

                    batch_results.append((dt_boxes, total_elapse / len(img_list)))
                return batch_results, total_elapse

            except:
                print("#################",img_list[0].shape, resize_w, resize_h)
                sys.exit(0)
        else:    
            for img in img_list:
                ori_im = img.copy()
                ori_imgs.append(ori_im)
                t_transform=time.time()
                data = {'image': img}
                data = transform(data, self.preprocess_op)
                print("t_transform",time.time()-t_transform)

                if data is None:
                    # 如果预处理失败，返回空结果
                    return [(None, 0) for _ in img_list], 0

                img_processed, shape_list = data
                batch_data.append(img_processed)
                batch_shapes.append(shape_list)
            # preprocess
            # t=time.time()
            print("preprocess",time.time()-t)
            # 堆叠成批处理张量
            t=time.time()

            try:
                
                batch_tensor = np.stack(batch_data, axis=0)
                batch_shapes = np.stack(batch_shapes, axis=0)
                batch_tensor = np.ascontiguousarray(batch_tensor)
                batch_shapes = np.ascontiguousarray(batch_shapes)
                print("stack",time.time()-t)
            except Exception as e:
                # 如果堆叠失败，回退到逐个处理
                batch_results = []
                for img in img_list:
                    dt_boxes, elapse = self.__call__(img)
                    batch_results.append((dt_boxes, elapse))
                return batch_results, time.time() - starttime

            # 批处理推理
            with torch.no_grad():
                inp = torch.from_numpy(batch_tensor)
                inp = inp.to(self.device)
                outputs = self.net(inp)
            # 处理输出
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
                t3=time.time()
                preds['maps'] = outputs['maps'].cpu().numpy()
                print("cpu_numpy",time.time()-t3) 
            elif self.det_algorithm == 'FCE':
                for i, (k, output) in enumerate(outputs.items()):
                    preds['level_{}'.format(i)] = output.cpu().numpy()
            else:
                raise NotImplementedError

            # 后处理每个图像的结果
            batch_results = []
            total_elapse = time.time() - starttime
            t4=time.time()

            for i in range(len(img_list)):
                # 提取单个图像的预测结果
                single_preds = {}
                for key, value in preds.items():
                    if isinstance(value, np.ndarray):
                        single_preds[key] = value[i:i + 1]  # 保持批次维度
                    else:
                        single_preds[key] = value

                # 后处理
                post_result = self.postprocess_op(single_preds, batch_shapes[i:i + 1])
                dt_boxes = post_result[0]['points']

                # 过滤和裁剪检测框
                dt_boxes = self._filter_det_res(dt_boxes, ori_imgs[i].shape)

                batch_results.append((dt_boxes, total_elapse / len(img_list)))
            print("后处理每个图像的结果",time.time()-t4) 
            return batch_results, total_elapse

    def batch_predict(self, img_list, max_batch_size=8):
        """
        批处理预测方法，支持多张图像同时检测

        Args:
            img_list: 图像列表
            max_batch_size: 最大批处理大小

        Returns:
            batch_results: 批处理结果列表，每个元素为(dt_boxes, elapse)
        """
        if not img_list:
            return []

        batch_results = []

        # 分批处理
        for i in range(0, len(img_list), max_batch_size):
            batch_imgs = img_list[i:i + max_batch_size]
            # assert尺寸一致
            t5 = time.time()
            batch_dt_boxes, batch_elapse = self._batch_process_same_size(batch_imgs)
            print("_batch_process_same_size",time.time()-t5) 

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

        with torch.no_grad():
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
            preds['maps'] = outputs['maps'].cpu().numpy()
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
