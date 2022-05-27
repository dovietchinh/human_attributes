import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import tensorrt as trt

# logger to capture errors, warnings, and other information during the build and inference phases
TRT_LOGGER = trt.Logger()

def build_engine(onnx_file_path):
    # initialize TensorRT engine and parse ONNX model
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # parse ONNX
    with open(onnx_file_path, 'rb') as model:
        print('Beginning ONNX file parsing')
        parser.parse(model.read())
    print('Completed parsing of ONNX file')
    # allow TensorRT to use up to 1GB of GPU memory for tactic selection
    # builder.max_workspace_size = 1 << 30
    # we have only one image in batch
    builder.max_batch_size = 1
    # use FP16 mode if possible
    if builder.platform_has_fast_fp16:
        builder.fp16_mode = True
    # generate TensorRT engine optimized for the target platform
    print('Building an engine...')
    engine = builder.build_cuda_engine(network)
    context = engine.create_execution_context()
    print("Completed creating Engine")

    return engine, context
def main():
    # initialize TensorRT engine and parse ONNX model
    ONNX_FILE_PATH = "/u01/Intern/chinhdv/code/multi-task-classification/result/runs_human_attributes_15/resnet50.onnx"
    engine, context = build_engine(ONNX_FILE_PATH)
    exit()
    # get sizes of input and output and allocate memory required for input data and for output data
    for binding in engine:
        if engine.binding_is_input(binding):  # we expect only one input
            input_shape = engine.get_binding_shape(binding)
            input_size = trt.volume(input_shape) * engine.max_batch_size * np.dtype(np.float32).itemsize  # in bytes
            device_input = cuda.mem_alloc(input_size)
        else:  # and one output
            output_shape = engine.get_binding_shape(binding)
            # create page-locked memory buffers (i.e. won't be swapped to disk)
            host_output = cuda.pagelocked_empty(trt.volume(output_shape) * engine.max_batch_size, dtype=np.float32)
            device_output = cuda.mem_alloc(host_output.nbytes)

    stream = cuda.Stream()

def main():
    import torch
    from torch2trt import torch2trt
    from torchvision.models.alexnet import alexnet

    # create some regular pytorch model...
    model = alexnet(pretrained=True).eval().cuda()

    # create example data
    x = torch.ones((1, 3, 224, 224)).cuda()

    # convert to TensorRT feeding sample data as input
    model_trt = torch2trt(model, [x])


if __name__ =='__main__':
    main()