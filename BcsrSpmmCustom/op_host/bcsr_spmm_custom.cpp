
#include "bcsr_spmm_custom_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

constexpr uint32_t MAX_MMAD_N = 32;

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    BcsrSpmmCustomTilingData tiling;
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());

    auto shape_a_addr = context->GetInputTensor(0)->GetData<int64_t>();
    auto shape_b = context->GetInputTensor(4)->GetOriginShape();

    auto shape_c = context->GetOutputShape(0)->GetOriginShape();
    int32_t M = shape_c.GetDim(0);
    int32_t K = shape_b.GetDim(0);
    int32_t N = shape_b.GetDim(1);

    tiling.set_M(M); 
    tiling.set_N(N);
    tiling.set_K(K);

    uint32_t totalLength = context->GetInputShape(1)->GetOriginShape().GetShapeSize() - 1;
    tiling.set_totalLength(totalLength);

    // core_info shape is [usedCoreNum * 4]
    uint32_t coreInfoSize = context->GetInputShape(5)->GetOriginShape().GetShapeSize();
    uint32_t usedCoreNum = coreInfoSize / 4;

    uint32_t maxCoreNum = ascendcPlatform.GetCoreNumAic();
    if (usedCoreNum > maxCoreNum) {
        usedCoreNum = maxCoreNum;
    }

    tiling.set_usedCoreNum(usedCoreNum);
    context->SetBlockDim(usedCoreNum);

    uint32_t alignNum = 32 / sizeof(uint16_t);

    uint32_t lastKLength = K % alignNum;
    if (lastKLength == 0) {
        lastKLength = alignNum;
    }
    tiling.set_lastKLength(lastKLength);

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    size_t *currentWorkspace = context->GetWorkspaceSizes(1);
    currentWorkspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    auto a_shape_addr = context->GetInputTensor(0)->GetData<int64_t>();
    auto b_shape = context->GetInputShape(4);
    auto c_shape = context->GetOutputShape(0);
    if (b_shape == nullptr || a_shape_addr == nullptr || c_shape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    int M = a_shape_addr[0];
    int N = b_shape->GetDim(1);
    c_shape->SetDimNum(2);
    c_shape->SetDim(0, M);
    c_shape->SetDim(1, N);

    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    if (context->SetOutputDataType(0, ge::DataType::DT_FLOAT) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class BcsrSpmmCustom : public OpDef {
public:
    explicit BcsrSpmmCustom(const char* name) : OpDef(name)
    {
        this->Input("a_shape")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .ValueDepend(REQUIRED);
        this->Input("row_ptr")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Input("col")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Input("val")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("b")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16})
            .Format({ge::FORMAT_ND});
        this->Input("core_info")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND});
        this->Output("c")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(BcsrSpmmCustom);
}
