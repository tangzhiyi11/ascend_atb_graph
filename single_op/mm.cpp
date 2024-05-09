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
    int ret = aclInit(nullptr);
    if (ret != 0) {
        std::cout << "aclInit failed, ret: " << ret << std::endl;
    }
    ret = aclrtSetDevice(0);
    if (ret != 0) {
       std::cout << "aclrtSetDevice failed, ret: " << ret << std::endl;
    }

    atb::infer::MatmulParam mm_param;
    // mm_param.transposeA = false;
    // mm_param.transposeB = false;

    atb::Operation *mm_op = nullptr;
    atb::Status st = atb::CreateOperation(mm_param, &mm_op);
    if (st != 0) {
        std::cout << "atb CreateOperation st != 0." << std::endl;
    }

    atb::VariantPack variant_pack;
    std::vector<int64_t> a_shape {1, 4096};
    std::vector<int64_t> b_shape {4096, 4096};
    std::vector<int64_t> out_shape {1, 4096};
    auto a_data = trans_to_fp16(get_random_fp32_data(a_shape));
    auto b_data = trans_to_fp16(get_random_fp32_data(b_shape));
    auto out_data = trans_to_fp16(get_random_fp32_data(out_shape));

    std::cout << "before compute!!" << std::endl;
    print_vector(trans_to_fp32(a_data), "a");
    print_vector(trans_to_fp32(b_data), "b");
    print_vector(trans_to_fp32(out_data), "out");
    std::cout << std::endl;

    auto a = genTensor(a_shape, ACL_FLOAT16, ACL_FORMAT_ND, a_data.data(), nullptr);
    auto b = genTensor(b_shape, ACL_FLOAT16, ACL_FORMAT_ND, b_data.data(), nullptr);
    auto out = genTensor(out_shape, ACL_FLOAT16, ACL_FORMAT_ND, out_data.data(), nullptr);

    variant_pack.inTensors.push_back(a);
    variant_pack.inTensors.push_back(b);
    variant_pack.outTensors.push_back(out);

    uint64_t workspaceSize = 0;
    st = mm_op->Setup(variant_pack, workspaceSize);
    if (st == 0) {
        std::cout << "mm op work space size: " << workspaceSize << std::endl;
    } else {
        std::cout << "mm op setup failed, st: " << st << std::endl;
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

    mm_op->Execute(variant_pack, static_cast<uint8_t*>(workspace), workspaceSize, context);

    ret = aclrtMemcpy(out_data.data(), out.dataSize, out.deviceData, out.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != 0) {
        std::cout << "tensor aclrtMemcpy failed, ret: " << ret << std::endl;
    }

    std::cout << "after compute !!"  << std::endl;
    print_vector(trans_to_fp32(out_data), "out");

    aclrtFree(a.deviceData);
    aclrtFree(b.deviceData);
    aclrtFree(out.deviceData);
    aclrtFree(workspace);

    ret = atb::DestroyContext(context);
    ret = aclrtDestroyStream(stream);
    st = atb::DestroyOperation(mm_op);

    return 0;
}