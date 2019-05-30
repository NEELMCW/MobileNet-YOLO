import sys
sys.path.append('../../python/')
import caffe
import re
import numpy as np
from collections import OrderedDict
from cfg import *
from prototxt import *

def reconvolution_counter():
    if 'cnt' not in reconvolution_counter.__dict__:
        reconvolution_counter.cnt = 0
    reconvolution_counter.cnt += 1
    return reconvolution_counter.cnt
def eltwise_counter():
    if 'cnt' not in eltwise_counter.__dict__:
        eltwise_counter.cnt = 0
    eltwise_counter.cnt += 1
    return eltwise_counter.cnt
def convolution_counter():
    if 'cnt' not in convolution_counter.__dict__:
        convolution_counter.cnt = 0
    convolution_counter.cnt += 1
    return convolution_counter.cnt
def bn_counter():
    if 'cnt' not in bn_counter.__dict__:
        bn_counter.cnt = 0
    bn_counter.cnt += 1
    return bn_counter.cnt
def scale_counter():
    if 'cnt' not in scale_counter.__dict__:
        scale_counter.cnt = 0
    scale_counter.cnt += 1
    return scale_counter.cnt
def relu_counter():
    if 'cnt' not in relu_counter.__dict__:
        relu_counter.cnt = 0
    relu_counter.cnt += 1
    return relu_counter.cnt
def route_counter():
    if 'cnt' not in route_counter.__dict__:
        route_counter.cnt = 0
    route_counter.cnt += 1
    return route_counter.cnt
def upsample_counter():
    if 'cnt' not in upsample_counter.__dict__:
        upsample_counter.cnt = 0
    upsample_counter.cnt += 1
    return upsample_counter.cnt







def darknet2caffe(cfgfile, weightfile, protofile, caffemodel):
    net_info = cfg2prototxt(cfgfile)
    save_prototxt(net_info , protofile, region=False)

    net = caffe.Net(protofile, caffe.TEST)
    params = net.params

    blocks = parse_cfg(cfgfile)
    fp = open(weightfile, "rb")  
    header = np.fromfile(fp, dtype = np.int32, count = 5)  
    buf = np.fromfile(fp, dtype = np.float32)
    fp.close()

    layers = []
    layer_id = 1
    start = 0
    for block in blocks:
        if start >= buf.size:
            break

        if block['type'] == 'net':
            continue
        elif block['type'] == 'convolutional':
            convCount = reconvolution_counter()
            batch_normalize = int(block['batch_normalize'])
            if 'name' in block.keys():  
                conv_layer_name = block['name']
                bn_layer_name = '%s-bn' % block['name']
                scale_layer_name = '%s-scale' % block['name']
            else:
                conv_layer_name = 'conv%d' % convCount
                bn_layer_name = 'conv%d_bn' % convCount
                scale_layer_name = 'conv%d_scale' % convCount

            if batch_normalize == 1:
                start = load_conv_bn2caffe(buf, start, params[conv_layer_name], params[bn_layer_name], params[scale_layer_name])
            else:
                start = load_conv2caffe(buf, start, params[conv_layer_name])
            layer_id = layer_id+1
        elif block['type'] == 'connected':
            if 'name' in block.keys():  
                fc_layer_name = block['name']
            else:
                fc_layer_name = 'layer%d-fc' % layer_id
            start = load_fc2caffe(buf, start, params[fc_layer_name])
            layer_id = layer_id+1
        elif block['type'] == 'maxpool':
            layer_id = layer_id+1
        elif block['type'] == 'avgpool':
            layer_id = layer_id+1
        elif block['type'] == 'region':
            layer_id = layer_id + 1
        elif block['type'] == 'route':
            layer_id = layer_id + 1
        elif block['type'] == 'shortcut':
            layer_id = layer_id + 1
        elif block['type'] == 'softmax':
            layer_id = layer_id + 1  
        elif block['type'] == 'cost':  
            layer_id = layer_id + 1  
        elif block['type'] == 'upsample':  
            layer_id = layer_id + 1
        elif block['type'] == 'yolo':
            layer_id = layer_id + 1
        else:
            print('unknow layer type %s ' % block['type'])
            layer_id = layer_id + 1
    print('save prototxt to %s' % protofile)
    #save_prototxt(net_info , protofile, region=True)
    print('save caffemodel to %s' % caffemodel)
    net.save(caffemodel)

def load_conv2caffe(buf, start, conv_param):
    weight = conv_param[0].data
    bias = conv_param[1].data
    conv_param[1].data[...] = np.reshape(buf[start:start+bias.size], bias.shape);   start = start + bias.size
    conv_param[0].data[...] = np.reshape(buf[start:start+weight.size], weight.shape); start = start + weight.size
    return start

def load_fc2caffe(buf, start, fc_param):
    weight = fc_param[0].data
    bias = fc_param[1].data
    fc_param[1].data[...] = np.reshape(buf[start:start+bias.size], bias.shape);   start = start + bias.size
    fc_param[0].data[...] = np.reshape(buf[start:start+weight.size], weight.shape); start = start + weight.size
    return start


def load_conv_bn2caffe(buf, start, conv_param, bn_param, scale_param):
    conv_weight = conv_param[0].data
    running_mean = bn_param[0].data
    running_var = bn_param[1].data
    scale_weight = scale_param[0].data
    scale_bias = scale_param[1].data

    scale_param[1].data[...] = np.reshape(buf[start:start+scale_bias.size], scale_bias.shape); start = start + scale_bias.size
    scale_param[0].data[...] = np.reshape(buf[start:start+scale_weight.size], scale_weight.shape); start = start + scale_weight.size
    bn_param[0].data[...] = np.reshape(buf[start:start+running_mean.size], running_mean.shape); start = start + running_mean.size
    bn_param[1].data[...] = np.reshape(buf[start:start+running_var.size], running_var.shape); start = start + running_var.size
    bn_param[2].data[...] = np.array([1.0])
    conv_param[0].data[...] = np.reshape(buf[start:start+conv_weight.size], conv_weight.shape); start = start + conv_weight.size
    return start

def cfg2prototxt(cfgfile):
    blocks = parse_cfg(cfgfile)

    layers = []
    props = OrderedDict() 
    bottom = 'data'
    layer_id = 1
    topnames = dict()
    yolo_count = 0
    mask = []
    bottom_yolo = []
    anchors_scale = []
    scale = 1
    num_out = 0 
    prev_layer = ""
    for block in blocks:
        if block['type'] == 'net':
            props['name'] = 'Yolov3_deploy'
            input_layer = OrderedDict()
            input_layer['name'] = "data"
            input_layer['type'] = "Input"
            input_layer['top'] = "data"
            input_param = OrderedDict()
            shape = OrderedDict()
            input_param["shape"] = shape 
            shape['dim'] = ['1']
            shape['dim'].append(block['channels'])
            shape['dim'].append(block['height'])
            shape['dim'].append(block['width'])
            input_layer['input_param'] = input_param
            layers.append(input_layer)
            continue
        elif block['type'] == 'convolutional':
            conv_layer = OrderedDict()
            if block.has_key('name'):
                conv_layer['name'] = block['name']
                conv_layer['type'] = 'Convolution'
                conv_layer['bottom'] = bottom + "_scale"
                conv_layer['top'] = block['name']
            else:
                convCount = convolution_counter()
                conv_layer['name'] = 'conv%d' % convCount
                conv_layer['type'] = 'Convolution'
                conv_layer['bottom'] = bottom
                if(re.match(r"conv",bottom)):
                        conv_layer['bottom'] = bottom + "_scale"
                else:
                        conv_layer['bottom'] = bottom 
                conv_layer['top'] = conv_layer['name']
            convolution_param = OrderedDict()
            #if layer_id>2 and int(block['size']) == 3 :
            #    convolution_param['group'] = num_out
            #    conv_layer['type'] = 'DepthwiseConvolution'
            #    convolution_param['engine'] = 1
            #else :
            #    conv_layer['type'] = 'Convolution'
            convolution_param['num_output'] = block['filters']
            if block['batch_normalize'] == '1':
                convolution_param['bias_term'] = 'false'
            if block['pad'] == '1':
                convolution_param['pad'] = str(int(block['size'])/2)
                convolution_param['pad'] = str(int(1))
            if block['size'] == '1':
                convolution_param['pad'] = 0
            convolution_param['kernel_size'] = block['size']
            #weight_filler_param = OrderedDict()
            #weight_filler_param['type'] = 'xavier'       
            #convolution_param['weight_filler'] = weight_filler_param
            #bias_filler_param = OrderedDict()
            #bias_filler_param['type'] = 'constant'       
            #convolution_param['bias_filler'] = bias_filler_param
            convolution_param['stride'] = block['stride']
            if int(block['stride'])==2:
                scale = scale * 2
            conv_layer['convolution_param'] = convolution_param
            layers.append(conv_layer)
            bottom = conv_layer['top']
            num_out = int(block['filters'])
            if block['batch_normalize'] == '1':
                bn_layer = OrderedDict()
                if block.has_key('name'):
                    bn_layer['name'] = '%s-bn' % block['name']
                else:
                    bn_layer['name'] = 'conv%d_bn' % convCount
                bn_layer['type'] = 'BatchNorm'
                bn_layer['bottom'] = bottom
                bn_layer['top'] = bn_layer['name']
                batch_norm_param = OrderedDict()
                batch_norm_param['use_global_stats'] = 'true'
                #batch_norm_param['eps'] = 0.0001
                bn_layer['batch_norm_param'] = batch_norm_param
                layers.append(bn_layer)

                scale_layer = OrderedDict()
                if block.has_key('name'):
                    scale_layer['name'] = '%s-scale' % block['name']
                else:
                    scale_layer['name'] = 'conv%d_scale' % convCount
                scale_layer['type'] = 'Scale'
                scale_layer['bottom'] = bn_layer['name']
                scale_layer['top'] = scale_layer['name']
                scale_param = OrderedDict()
                scale_param['bias_term'] = 'true'
                #filler_param = OrderedDict()
                #filler_param['value'] = 1.0
               # bias_filler_param = OrderedDict()
               # bias_filler_param['value'] = 0.0
               # scale_param['filler'] = filler_param
               # scale_param['bias_filler'] = bias_filler_param
                scale_layer['scale_param'] = scale_param
                layers.append(scale_layer)

            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                if block.has_key('name'):
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'relu%d' % convCount
                relu_layer['type'] = 'ReLU'
                relu_layer['bottom'] = scale_layer['name']
                relu_layer['top'] = scale_layer['name']
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        elif block['type'] == 'maxpool':
            max_layer = OrderedDict()
            max_layer['bottom'] = bottom
            if block.has_key('name'):
                max_layer['top'] = block['name']
                max_layer['name'] = block['name']
            else:
                max_layer['top'] = 'layer%d-maxpool' % layer_id
                max_layer['name'] = 'layer%d-maxpool' % layer_id
            max_layer['type'] = 'Pooling'
            pooling_param = OrderedDict()
            pooling_param['kernel_size'] = block['size']
            pooling_param['stride'] = block['stride']

            pooling_param['pool'] = 'MAX'
            pooling_param['pad'] = str((int(block['size'])-1)/2)
            if block.has_key('pad') and int(block['pad']) == 1:
                pooling_param['pad'] = str((int(block['size'])-1)/2)
            #if int(block['stride']) == 1 :
            #    pooling_param['pad'] = 0
            max_layer['pooling_param'] = pooling_param
            layers.append(max_layer)
            bottom = max_layer['top']
            topnames[layer_id] = bottom
            if int(block['stride']) == 2 :
                scale = scale * 2
            layer_id = layer_id+1
        elif block['type'] == 'avgpool':
            avg_layer = OrderedDict()
            avg_layer['bottom'] = bottom
            if block.has_key('name'):
                avg_layer['top'] = block['name']
                avg_layer['name'] = block['name']
            else:
                avg_layer['top'] = 'layer%d-avgpool' % layer_id
                avg_layer['name'] = 'layer%d-avgpool' % layer_id
            avg_layer['type'] = 'Pooling'
            pooling_param = OrderedDict()
            #pooling_param['kernel_size'] = 7
            #pooling_param['stride'] = 1
            pooling_param['pool'] = 'AVE'
            pooling_param['global_pooling'] = 'true'
            avg_layer['pooling_param'] = pooling_param
            layers.append(avg_layer)
            bottom = avg_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        elif block['type'] == 'region':
            if True:
                region_layer = OrderedDict()
                region_layer['bottom'] = bottom
                if block.has_key('name'):
                    region_layer['top'] = block['name']
                    region_layer['name'] = block['name']
                else:
                    region_layer['top'] = 'layer%d-region' % layer_id
                    region_layer['name'] = 'layer%d-region' % layer_id
                region_layer['type'] = 'Region'
                region_param = OrderedDict()
                region_param['anchors'] = block['anchors'].strip()
                region_param['classes'] = block['classes']
                region_param['num'] = block['num']
                region_layer['region_param'] = region_param
                layers.append(region_layer)
                bottom = region_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id + 1
        elif block['type'] == 'route':
            route_layer = OrderedDict()
            routeCount =  route_counter()
            layer_name = str(block['layers']).split(',')
            bottom_layer_dim = len(layer_name)
            if (bottom_layer_dim == 1):
                prev_layer_id = layer_id + int(block['layers'])  #(Added 1 for the miss yolo layer)
                bottom = topnames[prev_layer_id]
                if(re.match(r"conv",bottom)):
                	bottom = bottom + "_scale"
            if (bottom_layer_dim == 2):
                layer_name = [layer_id + int(idx) if int(idx) < 0 else int(idx) + 1 for idx in layer_name ]
                prev_layer_id1 = int(layer_name[0])
                prev_layer_id2 = int(layer_name[1])
                bottom1 = topnames[prev_layer_id1]
                bottom2 = topnames[prev_layer_id2]
                bottom = [bottom1, bottom2]
            if (bottom_layer_dim == 4):
                layer_name = [layer_id + int(idx) if int(idx) < 0 else int(idx) + 1 for idx in layer_name ]
                prev_layer_id1 = int(layer_name[0])
                prev_layer_id2 = int(layer_name[1])
                prev_layer_id3 = int(layer_name[2])
                prev_layer_id4 = int(layer_name[3])
                bottom1 = topnames[prev_layer_id1]
                bottom2 = topnames[prev_layer_id2]
                bottom3 = topnames[prev_layer_id3]
                bottom4 = topnames[prev_layer_id4]
                bottom = [bottom1, bottom2, bottom3, bottom4]
                #concat_param = OrderedDict()
                #concat_param['axis'] = '0'
                #route_layer['concat_param'] = concat_param
            if 'name' in block.keys():
                route_layer['name'] = block['name']
                if prev_layer == "yolo":
                    route_layer['type'] = 'Dropout'
                else:
                    route_layer['type'] = 'Concat'
                route_layer['bottom'] = bottom
                route_layer['top'] = block['name']
            else:
                route_layer['name'] = 'route%d' % routeCount
                if prev_layer == "yolo":
                    route_layer['type'] = 'Dropout'
                else:
                    route_layer['type'] = 'Concat'
                route_layer['bottom'] = bottom
                route_layer['top'] = 'route%d' % routeCount
            if prev_layer == "yolo":
                dropout_param = OrderedDict()
                dropout_param["dropout_ratio"] = 0.1
                route_layer["dropout_param"] = dropout_param
            else:
                concat_param = OrderedDict()
                concat_param["axis"] = 1
                route_layer["concat_param"] = concat_param
	    # Add constrainnt to avoid dangling route layer
            if(not(bottom_layer_dim == 1 and int(block['layers']) == -1)):
            	layers.append(route_layer)
            bottom = route_layer['top']
            topnames[layer_id] = bottom
            layer_id = layer_id + 1
            prev_layer = "route"
        elif block['type'] == 'upsample':
            upsampleCount = upsample_counter()
            upsample_layer = OrderedDict()
            print(block['stride'])
            if block.has_key('name'):
                upsample_layer['top'] = block['name']
                upsample_layer['name'] = block['name']
                upsample_layer['bottom'] = bottom + "_scale"
            else:
                upsample_layer['name'] = 'upsample%d' % upsampleCount 
                upsample_layer['type'] = 'Deconvolution'
                upsample_layer['bottom'] = bottom + "_scale"
                upsample_layer['top'] = 'upsample%d' % upsampleCount
            convolution_param = OrderedDict()
            convolution_param['num_output'] = num_out
            convolution_param['group'] = num_out
            convolution_param['kernel_size'] = block['stride']
            convolution_param['stride'] = block['stride']
            convolution_param['pad'] = 0 
            convolution_param['bias_term'] = 'false' 
            weight_filler = OrderedDict()
            weight_filler['type'] = "bilinear"
            convolution_param['weight_filler:'] = weight_filler
            param = OrderedDict()
            param['lr_mult'] = 0
            param['decay_mult'] = 0
            upsample_layer['convolution_param'] = convolution_param
            upsample_layer['param'] = param
            print(upsample_layer)
            layers.append(upsample_layer)
            bottom = upsample_layer['top']
            print('upsample:',layer_id)
            topnames[layer_id] = bottom
            scale = scale /2
            layer_id = layer_id + 1 
            prev_layer = "upsample"         
        elif block['type'] == 'yolo':
            
            anchor_len = len(block['anchors'].split(','))/2
            for i in block['mask'].split(',') :
                mask.append(i)
            #bottom_layer_dim = bottom['num_output']
            #print(scale) 
            #print(anchor_len)
            anchors_scale.append(scale)
            if len(mask)<anchor_len :
                yolo_layer = OrderedDict()
                yolo_layer['bottom'] = bottom
                yolo_layer['type'] = 'Concat'
                if 'name' in block.keys():
                    yolo_layer['top'] = block['name']
                    yolo_layer['name'] = block['name']
                else:
                    yolo_layer['top'] = 'layer%d-yolo' % layer_id
                    yolo_layer['name'] = 'layer%d-yolo' % layer_id
                #layers.append(yolo_layer)
                bottom = yolo_layer['top']
                topnames[layer_id] = bottom
                layer_id = layer_id + 1 
                bottom_yolo.append(yolo_layer['top'])
            else :
                yolo_layer = OrderedDict()
                bottom_yolo.append(bottom)
                yolo_layer['bottom'] = bottom_yolo
                yolo_layer['type'] = 'Yolov3DetectionOutput'
                if 'name' in block.keys():
                    yolo_layer['top'] = block['name']
                    yolo_layer['name'] = block['name']
                else:
                    yolo_layer['top'] = 'layer%d-yolo' % layer_id
                    yolo_layer['name'] = 'layer%d-yolo' % layer_id
                yolov3_detection_output_param = OrderedDict()
                yolov3_detection_output_param['nms_threshold']=0.45
                yolov3_detection_output_param['num_classes']=block['classes']
                yolov3_detection_output_param['biases'] = block['anchors'].split(',')
                yolov3_detection_output_param['mask'] = mask
                yolov3_detection_output_param['mask_group_num'] = yolo_count+1
                yolov3_detection_output_param['anchors_scale'] = anchors_scale
                yolo_layer['yolov3_detection_output_param'] = yolov3_detection_output_param
                #layers.append(yolo_layer)
                bottom = yolo_layer['top']
                topnames[layer_id] = bottom
                layer_id = layer_id + 1    
            yolo_count = yolo_count + 1
            prev_layer = "yolo"
        elif block['type'] == 'shortcut':
            prev_layer_id1 = layer_id + int(block['from'])
            prev_layer_id2 = layer_id - 1
            bottom1 = topnames[prev_layer_id1]
            bottom2= topnames[prev_layer_id2]
            if (not re.match(r"shortcut_eltwise",bottom1)):
                bottom1 = bottom1 + "_scale"
            if (not re.match(r"shortcut_eltwise",bottom2)):
                bottom2= bottom2 + "_scale"
            shortcut_layer = OrderedDict()
            if block.has_key('name'):
                shortcut_layer['name'] = block['name']
                shortcut_layer['type'] = 'Eltwise'
                shortcut_layer['bottom'] = [bottom1, bottom2]
                shortcut_layer['top'] = block['name']
            else:
                shortcut_layer['name'] = 'shortcut_eltwise%d' % eltwise_counter()
                shortcut_layer['type'] = 'Eltwise'
                shortcut_layer['bottom'] = [bottom1, bottom2]
                shortcut_layer['top'] = shortcut_layer['name'] 
            #eltwise_param = OrderedDict()
            #eltwise_param['operation'] = 'SUM'
            #shortcut_layer['eltwise_param'] = eltwise_param
            layers.append(shortcut_layer)
            bottom = shortcut_layer['top']
 
            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if block.has_key('name'):
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d-act' % layer_id
                relu_layer['type'] = 'ReLU'
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1                
        elif block['type'] == 'connected':
            fc_layer = OrderedDict()
            fc_layer['bottom'] = bottom
            if block.has_key('name'):
                fc_layer['top'] = block['name']
                fc_layer['name'] = block['name']
            else:
                fc_layer['top'] = 'layer%d-fc' % layer_id
                fc_layer['name'] = 'layer%d-fc' % layer_id
            fc_layer['type'] = 'InnerProduct'
            fc_param = OrderedDict()
            fc_param['num_output'] = int(block['output'])
            fc_layer['inner_product_param'] = fc_param
            layers.append(fc_layer)
            bottom = fc_layer['top']

            if block['activation'] != 'linear':
                relu_layer = OrderedDict()
                relu_layer['bottom'] = bottom
                relu_layer['top'] = bottom
                if block.has_key('name'):
                    relu_layer['name'] = '%s-act' % block['name']
                else:
                    relu_layer['name'] = 'layer%d-act' % layer_id
                relu_layer['type'] = 'ReLU'
                if block['activation'] == 'leaky':
                    relu_param = OrderedDict()
                    relu_param['negative_slope'] = '0.1'
                    relu_layer['relu_param'] = relu_param
                layers.append(relu_layer)
            topnames[layer_id] = bottom
            layer_id = layer_id+1
        else:
            print('unknow layer type %s ' % block['type'])
            topnames[layer_id] = bottom
            layer_id = layer_id + 1

    net_info = OrderedDict()
    net_info['props'] = props
    net_info['layers'] = layers
    return net_info

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 5:
        print('try:')
        print('python darknet2caffe.py tiny-yolo-voc.cfg tiny-yolo-voc.weights tiny-yolo-voc.prototxt tiny-yolo-voc.caffemodel')
        print('')
        print('please add name field for each block to avoid generated name')
        exit()

    cfgfile = sys.argv[1]
    #net_info = cfg2prototxt(cfgfile)
    #print_prototxt(net_info)
    #save_prototxt(net_info, 'tmp.prototxt')
    weightfile = sys.argv[2]
    protofile = sys.argv[3]
    caffemodel = sys.argv[4]
    darknet2caffe(cfgfile, weightfile, protofile, caffemodel)
