from __future__ import print_function
import argparse
import struct
import numpy as np
import cv2 as cv
from openvino.inference_engine import IECore
import sys
import os
import time
import psutil
import tensorflow.compat.v1 as tf

parser = argparse.ArgumentParser()
parser.add_argument('--version', required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--height', type=int, required=True)
parser.add_argument('--tf', type=bool, required=False, default=False) # compare with TF
parser.add_argument('--tune_perf', type=bool, required=False,default=True) # perf tuning based on https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_CPU.html
parser.add_argument('--input', help="Required. Path to an image files", required=True, type=str, nargs="+")
args = parser.parse_args()

conf_threshold = 0.5

def param_to_string(metric):
    if isinstance(metric, (list, tuple)):
        return ", ".join([str(val) for val in metric])
    elif isinstance(metric, dict):
        str_param_repr = ""
        for k, v in metric.items():
            str_param_repr += "{}: {}\n".format(k, v)
        return str_param_repr
    else:
        return str(metric)

def query_openvino_devices(ie):
    print("Available devices:")
    for device in ie.available_devices:
        print("\tDevice: {}".format(device))
        print("\tMetrics:")
        for metric in ie.get_metric(device, "SUPPORTED_METRICS"):
            try:
              metric_val = ie.get_metric(device, metric)
              print("\t\t{}: {}".format(metric, param_to_string(metric_val)))
            except TypeError:
              print("\t\t{}: UNSUPPORTED TYPE".format(metric))

        print("\n\tDefault values for device configuration keys:")
        for cfg in ie.get_metric(device, "SUPPORTED_CONFIG_KEYS"):
            try:
              cfg_val = ie.get_config(device, cfg)
              print("\t\t{}: {}".format(cfg, param_to_string(cfg_val)))
            except TypeError:
              print("\t\t{}: UNSUPPORTED TYPE".format(cfg))


def set_openvino_config(ie):
    #query_openvino_devices(ie)
    '''
    ubuntu@vm:~/workspace$ lscpu
    Architecture:                    x86_64
    CPU op-mode(s):                  32-bit, 64-bit
    Byte Order:                      Little Endian
    Address sizes:                   39 bits physical, 48 bits virtual
    CPU(s):                          2
    On-line CPU(s) list:             0,1
    Thread(s) per core:              1
    Core(s) per socket:              2
    Socket(s):                       1
    NUMA node(s):                    1
    Vendor ID:                       GenuineIntel
    CPU family:                      6
    Model:                           142
    Model name:                      Intel(R) Core(TM) i7-8569U CPU @ 2.80GHz
    Stepping:                        10
    CPU MHz:                         2807.998
    BogoMIPS:                        5615.99
    Hypervisor vendor:               KVM
    Virtualization type:             full
    L1d cache:                       64 KiB
    L1i cache:                       64 KiB
    L2 cache:                        512 KiB
    L3 cache:                        16 MiB
    NUMA node0 CPU(s):               0,1
    '''

    # see https://docs.openvinotoolkit.org/latest/openvino_docs_IE_DG_supported_plugins_CPU.html

    '''
    Binds inference threads to CPU cores. 'YES' (default) binding option maps threads to cores - this works best for static/synthetic scenarios like benchmarks. The 'NUMA' binding is more relaxed, binding inference threads only to NUMA nodes, leaving further scheduling to specific cores to the OS. This option might perform better in the real-life/contended scenarios. Note that for the latency-oriented cases (number of the streams is less or equal to the number of NUMA nodes, see below) both YES and NUMA options limit number of inference threads to the number of hardware cores (ignoring hyper-threading) on the multi-socket machines.
    '''
    #ie.set_config({"CPU_BIND_THREAD" : "NUMA"}, "CPU")

    '''
    Specifies the number of threads that CPU plugin should use for inference. Zero (default) means using all (logical) cores
    '''
    ie.set_config({"CPU_THREADS_NUM" : "0"}, "CPU")

    '''
    Specifies number of CPU "execution" streams for the throughput mode. Upper bound for the number of inference requests that can be executed simultaneously. All available CPU cores are evenly distributed between the streams. The default value is 1, which implies latency-oriented behavior for single NUMA-node machine, with all available cores processing requests one by one. On the multi-socket (multiple NUMA nodes) machine, the best latency numbers usually achieved with a number of streams matching the number of NUMA-nodes.
KEY_CPU_THROUGHPUT_NUMA creates as many streams as needed to accommodate NUMA and avoid associated penalties.
KEY_CPU_THROUGHPUT_AUTO creates bare minimum of streams to improve the performance; this is the most portable option if you don't know how many cores your target machine has (and what would be the optimal number of streams). Note that your application should provide enough parallel slack (for example, run many inference requests) to leverage the throughput mode.
Non-negative integer value creates the requested number of streams. If a number of streams is 0, no internal streams are created and user threads are interpreted as stream master threads.
    '''
    ie.set_config({"CPU_THROUGHPUT_STREAMS" : "CPU_THROUGHPUT_NUMA"}, "CPU")

    # lower inference precision - not supported
    #ie.set_config({"KEY_ENFORCE_BF16" : "YES"}, "CPU")

    '''
    These are general options, also supported by other plugins
    PARAMETER NAME	PARAMETER VALUES	DEFAULT	DESCRIPTION
KEY_EXCLUSIVE_ASYNC_REQUESTS	YES/NO	NO	Forces async requests (also from different executable networks) to execute serially. This prevents potential oversubscription
KEY_PERF_COUNT	YES/NO	NO	Enables gathering performance counter
    '''

    #ie_core.set_config({"PERF_COUNT": "YES"}, "CPU")

def dump_openvino_network_info(exec_net, ie):
    print("OpenVINO Network Info")
    input_layer = next(iter(exec_net.inputs))
    output_layer = next(iter(exec_net.outputs))
    input_shape = exec_net.inputs[input_layer].shape
    output_shape = exec_net.outputs[output_layer].shape
    print("Input layer {}".format(input_layer))
    print("Output layer {}".format(output_layer))
    print("Input shape {}".format(input_layer))
    print("Available Devices: {}".format(ie.available_devices))


def run_openvino_inference():
    tic = time.time()

    print("Reading IR model")
    ie = IECore()
    net = ie.read_network('efficientdet-{}_frozen.xml'.format(args.version),
                          'efficientdet-{}_frozen.bin'.format(args.version))

    print("Checking no of inputs and outputs")
    if len(net.input_info) == 0:
        raise AttributeError('No inputs info is provided')
    elif len(net.input_info) != 1:
        raise AttributeError("only one input layer network is supported")
    assert len(net.outputs) == 1, "Support only single output topologies"

    print("Getting input name and output blob")
    input_name = next(iter(net.input_info))
    output_blob =  next(iter(net.outputs))

    print("Tuning performance parameters")
    if args.tune_perf:
        set_openvino_config(ie)

    print("Setting network batch size to {}".format(len(args.input)))
    net.batch_size = len(args.input)

    # download network onto the device
    print("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name='CPU')

    # dump network info
    # print("Dumping networking info")
    #dump_openvino_network_info(exec_net, ie)
   
    print("Preparing inputs")
    if len(net.input_info[input_name].input_data.shape) != 4:
        raise AttributeError("Incorrect shape {} for network".format(args.shape))
    n, c, h, w = net.input_info[input_name].input_data.shape
    print(f"Batch size is {n}")

    # Read and pre-process input images
    print("Reading images")
    images = np.ndarray(shape=(n, c, h, w))
    images_orig = []
    for i in range(n):
        image = cv.imread(args.input[i])
        images_orig.append(image)
        if image.shape[:-1] != (h, w):
            print(f"Image {args.input[i]} is resized from {image.shape[:-1]} to {(h, w)}")
            image = cv.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image

    # Call inference passing in dict input_name => images
    print("Running inference")
    tic1 = time.time()
    output_result  = exec_net.infer({input_name : images})
    toc1 = time.time()
    print("elapsed time: inference: {}".format(toc1-tic1))

    #output_result = next(iter(output_result.values()))
    #output_result = output_result.reshape(-1, 7)
    output_result = output_result[output_blob]
    data = output_result[0][0]

    print('\nOpenVINO predictions')
    print("confidence, xmin, ymin, xmax, ymax")
    img_id_list = []
    # for number, detection in enumerate(output_result):
    for number, detection in enumerate(data):
        conf = detection[2]
        if conf < conf_threshold:
            continue
        img_id = np.int(detection[0])
        img_h, img_w = images[img_id].shape[1], images[img_id].shape[2]
        print("img_h = {}".format(img_h))
        print("img_w = {}".format(img_w))
        #print("detection = {}".format(detection))
        img_label = np.int(detection[1])
        xmin = np.int(img_w * detection[3])
        ymin = np.int(img_h * detection[4])
        xmax = np.int(img_w * detection[5])
        ymax = np.int(img_h * detection[6])
        cv.rectangle(images_orig[img_id], (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=1)
        img_id_list.append(img_id)
        print(conf, xmin, ymin, xmax, ymax)

    toc = time.time()
    print("elapsed time: run_openvino_inference: {}".format(toc-tic))
    print("avg elapsed time: run_openvino_inference: {}".format((toc-tic)/float(n)))

    tic = time.time()
    cv.imwrite('vikrant_batch_{}.png'.format(args.version), images_orig[img_id_list.pop()])
    toc = time.time()
    print("save_image: {}".format(toc-tic))

def run_tensorflow_inference():
    tic = time.time()
    pb_file = 'savedmodel_{}/efficientdet-{}_frozen.pb'.format(args.version, args.version)
    graph_def = tf.compat.v1.GraphDef()

    try:
        with tf.io.gfile.GFile(pb_file, 'rb') as f:
            graph_def.ParseFromString(f.read())
    except:
        with tf.gfile.FastGFile(pb_file, 'rb') as f:
            graph_def.ParseFromString(f.read())

    with tf.compat.v1.Session() as sess:
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        for node in graph_def.node:
            if node.name == 'Const_3':
                means = struct.unpack('fff', node.attr['value'].tensor.tensor_content)
                print('Mean values are', [m * 255 for m in means])

        tfOut = sess.run(sess.graph.get_tensor_by_name('detections:0'),
                         feed_dict={'image_arrays:0': np.expand_dims(img, axis=0)})

        # Render detections
        print('\nTensorFlow predictions')
        print("confidence, xmin, ymin, xmax, ymax")
        tfOut = tfOut.reshape(-1, 7)
        for detection in tfOut:
            conf = detection[5]
            if conf < conf_threshold:
                continue
            ymin, xmin, ymax, xmax = [int(v) for v in detection[1:5]]
            cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 127, 255), thickness=3)
            print(conf, xmin, ymin, xmax, ymax)
    toc = time.time()
    print("elapsed time: run_tensorflow_inference: {}".format(toc-tic))

def main():
    run_openvino_inference()

if __name__ == "__main__":
    main()
