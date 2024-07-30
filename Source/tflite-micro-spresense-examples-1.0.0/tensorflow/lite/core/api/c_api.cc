/****************************************************************************
 * c_api.cc
 *
 *   Copyright (C) 2020 Wilderness Labs. All rights reserved.
 *   Author: Wilderness Labs
 * 
 ****************************************************************************/

#include "tensorflow/lite/core/api/c_api.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/micro/spresense/debug_log_callback.h"
#include "c_api.h"

namespace
{
    tflite::AllOpsResolver resolver;
    static tflite::MicroErrorReporter micro_error_reporter;
}

extern "C"
{

    TfLiteMicroModelOption *TfLiteMicroGetModel(int arena_size, uint8_t *arena, const void *model_data)
    {
        auto model = tflite::GetModel(model_data);

        if (model->version() != TFLITE_SCHEMA_VERSION)
        {
            /**
             * Show message error or return some status
             */
            printf("[ERROR] - Model provied is shema version %ld\n", model->version());
            return nullptr;
        }
        static TfLiteMicroModelOption micro_option;
        micro_option.arena = arena;
        micro_option.arena_size = arena_size;
        micro_option.model = model;

        return &micro_option;
    }

    TfLiteMicroInterpreterOption *TfLiteMicroInterpreterOptionCreate(TfLiteMicroModelOption *option)
    {

        if (option == nullptr)
        {
            printf("[ERROR] - Failed to create an interpreter option\n");
            return nullptr;
        }

        static TfLiteMicroInterpreterOption micro_interpreter_option;

        micro_interpreter_option.error_reporter = &micro_error_reporter;

        return &micro_interpreter_option;
    }

    TfLiteMicroInterpreter *TfLiteMicroInterpreterCreate(TfLiteMicroInterpreterOption *interpreter_option, TfLiteMicroModelOption *model_option)
    {
        if (interpreter_option == nullptr || model_option == nullptr)
        {
            printf("[ERROR] Failed to create interpreter\n");
            return nullptr;
        }

        static TfLiteMicroInterpreter micro_interpreter;

        static tflite::MicroInterpreter interpreter(model_option->model,
                                                    resolver,
                                                    model_option->arena,
                                                    model_option->arena_size,
                                                    interpreter_option->error_reporter,
                                                    nullptr, nullptr);
        micro_interpreter.interpreter = &interpreter;

        return &micro_interpreter;
    }

    void TfLiteMicroInterpreterOptionDelete(TfLiteMicroInterpreterOption *option)
    {
        if (option != NULL)
        {
            delete option;
        }
    }

    void TfLiteMicroModelDelete(TfLiteMicroModelOption *model)
    {
        if (model != NULL)
        {
            delete model;
        }
    }

    void TfLiteMicroInterpreterDelete(TfLiteMicroInterpreter *interpreter)
    {
        if (interpreter != NULL)
        {
            delete interpreter;
        }
    }

    TfLiteStatus TfLiteMicroInterpreterAllocateTensors(TfLiteMicroInterpreter *inter)
    {
        return inter->interpreter->AllocateTensors();
    }

    TfLiteTensor *TfLiteMicroInterpreterGetInput(const TfLiteMicroInterpreter *inter, int32_t index)
    {
        return inter->interpreter->input(index);
    }

    TfLiteTensor *TfLiteMicroInterpreterGetOutput(const TfLiteMicroInterpreter *inter, int32_t index)
    {
        return inter->interpreter->output(index);
    }

    int32_t TfLiteMicroInterpreterGetOutputSize(const TfLiteMicroInterpreter *inter)
    {
        return static_cast<int32_t>(inter->interpreter->outputs_size());
    }

    int32_t TfLiteMicroInterpreterGetInputSize(const TfLiteMicroInterpreter *inter)
    {
        return static_cast<int32_t>(inter->interpreter->inputs_size());
    }

    TfLiteStatus TfLiteMicroInterpreterInvoke(TfLiteMicroInterpreter *inter)
    {
        return inter->interpreter->Invoke();
    }

    TfLiteQuantizationParams TfLiteMicroTensorQuantizationParams(const TfLiteTensor *tensor)
    {
        return tensor->params;
    }

    /**
     * TODO: The tensor data have a large type variation, so it should be nice
     * return a void pointer and include one more parameter  to select which type you want (float, int8_t ...)
     */

    int8_t TfLiteMicroGeInt8tData(const TfLiteTensor *tensor, int index)
    {
        return tensor->data.int8[index];
    }

    void TfLiteMicroSetInt8Data(const TfLiteTensor *tensor, int index, int8_t value)
    {
        tensor->data.int8[index] = value;
    }

    void TfLiteMicroSetData(const TfLiteTensor *tensor, int index, void *value, TfLiteType type)
    {
        switch (type) 
        {
            case kTfLiteFloat32:
                tensor->data.f[index] = *((float*)value);
            break;

            case kTfLiteInt16:
                tensor->data.i16[index] = *((int16_t*)value);
            break;

            case kTfLiteInt32:
                tensor->data.i32[index] = *((int32_t*)value);
            break;

            case kTfLiteUInt32:
                tensor->data.u32[index] = *((uint32_t*)value);
            break;

            case kTfLiteUInt8:
                tensor->data.uint8[index] = *((uint8_t*)value);
            break;

            case kTfLiteInt8:
                tensor->data.int8[index] = *((int8_t*)value);
            break;

            case kTfLiteInt64:
                tensor->data.i64[index] = *((int64_t*)value);
            break;

            case kTfLiteUInt64:
                tensor->data.u64[index] = *((uint64_t*)value);
            break;

            case kTfLiteBool:
                tensor->data.b[index] = *((bool*)value);
            break;

            case kTfLiteComplex64:
                tensor->data.c64[index] = *((TfLiteComplex64*)value);
            break;

            case kTfLiteComplex128:
                tensor->data.c128[index] = *((TfLiteComplex128*)value);
            break;

            case kTfLiteString:
                tensor->data.raw[index] = *((char*)value);
            break;

            case kTfLiteFloat16:
                tensor->data.f16[index] = *((TfLiteFloat16*)value);
            break;

            case kTfLiteFloat64:
                tensor->data.f64[index] = *((double*)value);
            break;

            default:
            break;
        }
    }

    void *TfLiteMicroGetData(const TfLiteTensor *tensor, int index, TfLiteType type)
    {
        switch (type) 
        {
            case kTfLiteFloat32:
                return &tensor->data.f[index];

            case kTfLiteInt16:
                return &tensor->data.i16[index];
            break;

            case kTfLiteInt32:
                return &tensor->data.i32[index];
            break;

            case kTfLiteUInt32:
                return &tensor->data.u32[index];
            break;

            case kTfLiteUInt8:
                return &tensor->data.uint8[index];
            break;

            case kTfLiteInt8:
                return &tensor->data.int8[index];
            break;

            case kTfLiteInt64:
                return &tensor->data.i64[index];
            break;

            case kTfLiteUInt64:
                return &tensor->data.u64[index];
            break;

            case kTfLiteBool:
                return &tensor->data.b[index];
            break;

            case kTfLiteComplex64:
                return &tensor->data.c64[index];
            break;

            case kTfLiteComplex128:
                return &tensor->data.c128[index];
            break;

            case kTfLiteString:
                return &tensor->data.raw[index];
            break;

            case kTfLiteFloat16:
                return &tensor->data.f16[index];
            break;

            case kTfLiteFloat64:
                return &tensor->data.f64[index];
            break;

            default:
            return 0
            ;
        }

    }

    float TfLiteMicroGetFloatData(const TfLiteTensor *tensor, int index)
    {
        return tensor->data.f[index];
    }

    void TfLiteMicroSetFloatData(const TfLiteTensor *tensor, int index, float value)
    {
        tensor->data.f[index] = value;
    }

    int TfLiteMicroDimsData(const TfLiteTensor *tensor, int index)
    {
        return tensor->dims->data[index];
    }

    int TfLiteMicroDimsSizeData(const TfLiteTensor *tensor)
    {
        return tensor->dims->size;
    }

    TfLiteType TfLiteMicroGetType(const TfLiteTensor *tensor)
    {
        return tensor->type;
    }

    int TfLiteMicroGetByte(const TfLiteTensor *tensor)
    {
        return tensor->bytes;
    }
}