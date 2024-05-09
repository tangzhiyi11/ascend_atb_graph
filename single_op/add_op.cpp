#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "atb/atb_infer.h"

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
        // std::cout << "device_data ptr:" << device_data << std::endl;
        // std::cout << "host_data ptr:" << host_data << std::endl;
        // std::cout << "data_size:" << data_size << std::endl;
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

    atb::infer::ElewiseParam addParam;
    addParam.elewiseType =  atb::infer::ElewiseParam::ELEWISE_ADD;

    atb::Operation *addOp = nullptr;
    atb::Status st = atb::CreateOperation(addParam, &addOp);
    if (st != 0) {
        std::cout << "atb CreateOperation st != 0." << std::endl;
    }

    atb::VariantPack variant_pack;
    std::vector<float> x1_data {1, 2, 3, 4};
    std::vector<float> x2_data {5, 6, 7, 8};
    std::vector<float> out_data {0, 0, 0, 0};
    std::vector<int64_t> dims {2, 2};
    auto x1 = genTensor(dims, ACL_FLOAT, ACL_FORMAT_ND, x1_data.data(), nullptr);
    auto x2 = genTensor(dims, ACL_FLOAT, ACL_FORMAT_ND, x2_data.data(), nullptr);
    auto out = genTensor(dims, ACL_FLOAT, ACL_FORMAT_ND, out_data.data(), nullptr);

    variant_pack.inTensors.push_back(x1);
    variant_pack.inTensors.push_back(x2);
    variant_pack.outTensors.push_back(out);

    uint64_t workspaceSize = 0;
    st = addOp->Setup(variant_pack, workspaceSize);
    if (st == 0) {
        std::cout << "add op work space size: " << workspaceSize << std::endl;
    } else {
        std::cout << "add op setup failed, st: " << st << std::endl;
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

    addOp->Execute(variant_pack, static_cast<uint8_t*>(workspace), workspaceSize, context);

    ret = aclrtMemcpy(out_data.data(), out.dataSize, out.deviceData, out.dataSize, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != 0) {
        std::cout << "tensor aclrtMemcpy failed, ret: " << ret << std::endl;
    }

    std::cout << "after execute, out: ";
    for (const auto i : out_data) {
        std::cout << i << " ,"; 
    }
    std::cout << std::endl;

    aclrtFree(x1.deviceData);
    aclrtFree(x2.deviceData);
    aclrtFree(out.deviceData);
    aclrtFree(workspace);

    ret = atb::DestroyContext(context);
    ret = aclrtDestroyStream(stream);
    st = atb::DestroyOperation(addOp);

    return 0;
}