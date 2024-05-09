#include <iostream>
#include <random>
#include <vector>

#include "acl/acl.h"
#include "atb/atb_infer.h"

float get_random() {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(-1, 1);
    return dis(e);
}

std::vector<float> get_random_fp32_data(const std::vector<int64_t> shapes) {
    int64_t cnt = 1;
    for (auto i : shapes) {
        cnt *= i;
    }
    std::vector<float> res(cnt);
    for (int i = 0; i < cnt; ++i) {
        res[i] = get_random();
    }
    return res;
}

std::vector<aclFloat16> trans_to_fp16(const std::vector<float>& input) {
    std::vector<aclFloat16> res;
    for (unsigned int i = 0; i < input.size(); ++i) {
        res.push_back(aclFloatToFloat16(input[i]));
    }
    return res;
}
std::vector<float> trans_to_fp32(const std::vector<aclFloat16>& input) {
    std::vector<float> res;
    for (unsigned int i = 0; i < input.size(); ++i) {
        res.push_back(aclFloat16ToFloat(input[i]));
    }
    return res;
}

void print_vector(const std::vector<float>& input, const std::string& name) {
    std::cout << name << ": [ ";
    for (unsigned int i = 0; i < input.size() && i < 10; ++i) {
        std::cout << input[i] << ", ";
    }
    std::cout << " ]" << std::endl;
}

atb::Tensor genTensor(const std::vector<int64_t>& dims, aclDataType dtype, aclFormat format, void* host_data, void* device_data) {
    atb::Dims atb_dims;
    atb::TensorDesc desc;
    atb::Tensor tensor;

    // init atb dims
    auto dim_num = dims.size();
    atb_dims.dimNum = static_cast<uint64_t>(dim_num);
    int nums = 1;
    for (unsigned int i = 0; i < dim_num; ++i) {
        atb_dims.dims[i] = dims[i];
        nums *= dims[i];
    }
    int64_t data_size = nums * aclDataTypeSize(dtype);

    // init atb tensor desc
    desc.dtype = dtype;
    desc.format = format;
    desc.shape = atb_dims;

    // init tensor
    tensor.desc = desc;
    tensor.deviceData = device_data;
    tensor.dataSize = static_cast<uint64_t>(data_size);

    if (host_data && device_data == nullptr) {
        void* tmp = nullptr;
        int ret = aclrtMalloc(&tmp, data_size, ACL_MEM_MALLOC_HUGE_FIRST);
        if ( ret != 0) {
            std::cout << "tensor acltrMalloc failed, ret: " << ret << std::endl;
        }
        ret = aclrtMemcpy(tmp, data_size, host_data, data_size, ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != 0) {
            std::cout << "tensor aclrtMemcpy failed, ret: " << ret << std::endl;
        }
        tensor.deviceData = tmp;
    }

    return tensor;
}

int main() {
    std::cout << "this is a atb test program!!" << std::endl;

    int ret = aclInit(nullptr);
    if (ret != 0) {
        std::cout << "aclInit failed, ret: " << ret << std::endl;
    }
    ret = aclrtSetDevice(0);
    if (ret != 0) {
       std::cout << "aclrtSetDevice failed, ret: " << ret << std::endl;
    }

    // a1: 1, 4096 b1: 4096, 4096     a2: 1, 4096  b2: 4096, 4096
    //       \             /               \               /
    //         mm1: 1, 4096                    mm2: 1, 4096
    //               \                               /
    //                \                             /
    //                 \                           /
    //                  \                         /
    //                       add: 1, 4096 

    // mm1
    std::vector<int64_t> a1_shape {1, 4096};
    std::vector<int64_t> b1_shape {4096, 4096};
    std::vector<int64_t> mm1_shape {1, 4096};
    auto a1_data = trans_to_fp16(get_random_fp32_data(a1_shape));
    auto b1_data = trans_to_fp16(get_random_fp32_data(b1_shape));
    auto a1 = genTensor(a1_shape, ACL_FLOAT16, ACL_FORMAT_ND, a1_data.data(), nullptr);
    auto b1 = genTensor(b1_shape, ACL_FLOAT16, ACL_FORMAT_ND, b1_data.data(), nullptr);

    atb::infer::MatmulParam mm1_param;
    atb::Operation *mm1_op = nullptr;
    atb::Status st = atb::CreateOperation(mm1_param, &mm1_op);
    if (st != 0) {
        std::cout << "atb CreateOperation mm1 failed, st: " << st << std::endl;
    }

    // mm2
    std::vector<int64_t> a2_shape {1, 4096};
    std::vector<int64_t> b2_shape {4096, 4096};
    std::vector<int64_t> mm2_shape {1, 4096};
    auto a2_data = trans_to_fp16(get_random_fp32_data(a2_shape));
    auto b2_data = trans_to_fp16(get_random_fp32_data(b2_shape));
    auto a2 = genTensor(a2_shape, ACL_FLOAT16, ACL_FORMAT_ND, a2_data.data(), nullptr);
    auto b2 = genTensor(b2_shape, ACL_FLOAT16, ACL_FORMAT_ND, b2_data.data(), nullptr);

    atb::infer::MatmulParam mm2_param;
    atb::Operation *mm2_op = nullptr;
    st = atb::CreateOperation(mm2_param, &mm2_op);
    if (st != 0) {
        std::cout << "atb CreateOperation mm2 failed, st: " << st << std::endl;
    }

    // add
    std::vector<int64_t> out_shape {1, 4096};
    auto out_data = trans_to_fp16(get_random_fp32_data(out_shape));
    auto out = genTensor(out_shape, ACL_FLOAT16, ACL_FORMAT_ND, out_data.data(), nullptr);

    std::cout << "before compute!!" << std::endl;
    print_vector(trans_to_fp32(a1_data), "a1");
    print_vector(trans_to_fp32(b1_data), "b1");
    print_vector(trans_to_fp32(a2_data), "a2");
    print_vector(trans_to_fp32(b2_data), "b2");
    print_vector(trans_to_fp32(out_data), "out");
    std::cout << std::endl;

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType =  atb::infer::ElewiseParam::ELEWISE_ADD;
    atb::Operation *add_op = nullptr;
    st = atb::CreateOperation(addParam, &add_op);
    if (st != 0) {
        std::cout << "atb CreateOperation add failed, st: " << st << std::endl;
    }

    // graph
    atb::GraphParam graph_param;
    graph_param.inTensorNum = 4;
    graph_param.outTensorNum = 1;
    graph_param.internalTensorNum = 2;
    graph_param.nodes.resize(3);

    graph_param.nodes[0].operation = mm1_op;
    graph_param.nodes[0].inTensorIds = {0, 1};
    graph_param.nodes[0].outTensorIds = {5};
    graph_param.nodes[1].operation = mm2_op;
    graph_param.nodes[1].inTensorIds = {2, 3};
    graph_param.nodes[1].outTensorIds = {6};
    graph_param.nodes[2].operation = add_op;
    graph_param.nodes[2].inTensorIds = {5, 6};
    graph_param.nodes[2].outTensorIds = {4};

    atb::VariantPack variant_pack;
    variant_pack.inTensors.push_back(a1);  // 0
    variant_pack.inTensors.push_back(b1);  // 1
    variant_pack.inTensors.push_back(a2);  // 2
    variant_pack.inTensors.push_back(b2);  // 3
    variant_pack.outTensors.push_back(out); // 4

    atb::Operation *graph = nullptr;
    st = atb::CreateOperation(graph_param, &graph);
    if (st != 0) {
        std::cout << "atb CreateOperation graph failed, st: " << st << std::endl;
    }

    uint64_t workspaceSize = 0;
    st = graph->Setup(variant_pack, workspaceSize);
    if (st == 0) {
        std::cout << "graph work space size: " << workspaceSize << std::endl;
    } else {
        std::cout << "graph setup failed, st: " << st << std::endl;
    }

    void* workspace = nullptr;
    if (workspaceSize > 0) {
        ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != 0) {
            std::cout << "malloc workspace failed, ret: " << ret << std::endl;
        }
    }

    atb::Context *context = nullptr;
    ret = atb::CreateContext(&context);
    void *stream = nullptr;
    ret = aclrtCreateStream(&stream);
    context->SetExecuteStream(stream);

    graph->Execute(variant_pack, static_cast<uint8_t*>(workspace), workspaceSize, context);

    ret = aclrtMemcpy(out_data.data(), out.dataSize, out.deviceData, out.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != 0) {
        std::cout << "tensor aclrtMemcpy failed, ret: " << ret << std::endl;
    }

    std::cout << "after compute !!"  << std::endl;
    print_vector(trans_to_fp32(out_data), "out");

    aclrtFree(a1.deviceData);
    aclrtFree(a2.deviceData);
    aclrtFree(b1.deviceData);
    aclrtFree(b2.deviceData);
    aclrtFree(out.deviceData);
    aclrtFree(workspace);

    ret = atb::DestroyContext(context);
    ret = aclrtDestroyStream(stream);
    st = atb::DestroyOperation(graph);
    return 0;
}