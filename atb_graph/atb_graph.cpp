#include <iostream>
#include <random>
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
    tensor.hostData = host_data;
    tensor.deviceData = device_data;
    tensor.dataSize = static_cast<uint64_t>(data_size);
    return tensor;
}

class AtbGraph {
  public:
    explicit AtbGraph(void* _outter_workspace, void* outter_stream) : outter_workspace(_outter_workspace), stream(outter_stream) {
        int ret = atb::CreateContext(&context);
        if (ret != 0) {
            std::cout << "atb::CreateContext faield, ret: " << ret << std::endl;
        }
    
        if (stream == nullptr) {
            ret = aclrtCreateStream(&stream);
            if (ret != 0) {
                std::cout << "aclrtCreateStream faield, ret: " << ret << std::endl;
            }
        }
        context->SetExecuteStream(stream);

        if (outter_workspace != nullptr) {
            workspace = outter_workspace;
        }
        build();
    }

    void build() {
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
        auto a1 = genTensor(a1_shape, ACL_FLOAT16, ACL_FORMAT_ND, nullptr, nullptr);
        auto b1 = genTensor(b1_shape, ACL_FLOAT16, ACL_FORMAT_ND, nullptr, nullptr);

        atb::infer::MatmulParam mm1_param;
        mm1_param.transposeA = false;
        mm1_param.transposeB = false;
        atb::Operation *mm1_op = nullptr;
        atb::Status st = atb::CreateOperation(mm1_param, &mm1_op);
        if (st != 0) {
            std::cout << "atb CreateOperation mm1 failed, st: " << st << std::endl;
        }

        // mm2
        std::vector<int64_t> a2_shape {1, 4096};
        std::vector<int64_t> b2_shape {4096, 4096};
        auto a2 = genTensor(a2_shape, ACL_FLOAT16, ACL_FORMAT_ND, nullptr, nullptr);
        auto b2 = genTensor(b2_shape, ACL_FLOAT16, ACL_FORMAT_ND, nullptr, nullptr);

        atb::infer::MatmulParam mm2_param;
        mm2_param.transposeA = false;
        mm2_param.transposeB = false;
        atb::Operation *mm2_op = nullptr;
        st = atb::CreateOperation(mm2_param, &mm2_op);
        if (st != 0) {
            std::cout << "atb CreateOperation mm2 failed, st: " << st << std::endl;
        }

        // add
        std::vector<int64_t> out_shape {1, 4096};
        auto out = genTensor(out_shape, ACL_FLOAT16, ACL_FORMAT_ND, nullptr, nullptr);

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

        variant_pack.inTensors.push_back(a1);  // 0
        variant_pack.inTensors.push_back(b1);  // 1
        variant_pack.inTensors.push_back(a2);  // 2
        variant_pack.inTensors.push_back(b2);  // 3
        variant_pack.outTensors.push_back(out); // 4

        st = atb::CreateOperation(graph_param, &graph);
        if (st != 0) {
            std::cout << "atb CreateOperation graph failed, st: " << st << std::endl;
        }

        st = graph->Setup(variant_pack, workspaceSize);
        if (st == 0) {
            std::cout << "graph work space size: " << workspaceSize << std::endl;
        } else {
            std::cout << "graph setup failed, st: " << st << std::endl;
        }

        if (workspace == nullptr && workspaceSize > 0) {
            int ret = aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
            if (ret != 0) {
                std::cout << "malloc workspace failed, ret: " << ret << std::endl;
            }
        }
    }

    void run(void* inputs[], int input_size, void* outputs[], int output_size) {
        for (int i = 0; i < input_size; ++i) {
            variant_pack.inTensors[i].deviceData = inputs[i];
            // std::cout << "input ptr: " << inputs[i] << std::endl;
        }
        for (int i = 0; i < output_size; ++i) {
            variant_pack.outTensors[i].deviceData = outputs[i];
            // std::cout << "out ptr: " << outputs[i] << std::endl;
        }
        graph->Setup(variant_pack, workspaceSize);
        graph->Execute(variant_pack, static_cast<uint8_t*>(workspace), workspaceSize, context);
        aclrtSynchronizeStream(stream);
    }

    ~AtbGraph() {
        if (outter_workspace == nullptr && workspace != nullptr) {
            aclrtFree(workspace);
        } 

        int ret = atb::DestroyContext(context);
        if (ret != 0) {
            std::cout << "atb::DestroyContext faield, ret: " << ret << std::endl;
        }

        ret = aclrtDestroyStream(stream);
        if (ret != 0) {
            std::cout << "aclrtDestroyStream faield, ret: " << ret << std::endl;
        }

        atb::Status st = atb::DestroyOperation(graph);
        if (st != 0) {
            std::cout << "atb::DestroyOperation faield, st: " << st << std::endl;
        }
    }

  private:
    void *outter_workspace;
    void *stream;
    atb::VariantPack variant_pack;
    atb::Operation *graph;
    uint64_t workspaceSize;
    atb::Context *context;
    void *workspace;
};

AtbGraph* graph = nullptr;

extern "C" void init(void* workspace, void* stream) {
    graph = new AtbGraph(workspace, stream);
}

extern "C" void run(void* inputs[], int input_size, void* outputs[], int output_size) {
    graph->run(inputs, input_size, outputs, output_size);
}
