/****************************************************************************
 * c_api.cc
 *
 *   Copyright (C) 2024 Wilderness Labs. All rights reserved.
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
    /****************************************************************************
     * Name: TfLiteMicroGetModel
     *
     * Description:
     *    Load the model as a flat buffer and store the arena buffer.
     *
     * Parameters:
     *   arena_size : Size of the arena.
     *   arena      : Pointer to the arena.
     *   model_data : Pointer to hold the model.
     *
     * Returned Value:
     *  If the model corresponds to the current version of TF returns a pointer struct otherwise nullptr.
     *
     ****************************************************************************/
    TfLiteMicroModelOption *TfLiteMicroGetModel(int arena_size, uint8_t *arena, const void *model_data)
    {
        auto model = tflite::GetModel(model_data);

        if (model->version() != TFLITE_SCHEMA_VERSION)
        {
            return nullptr;
        }
        static TfLiteMicroModelOption micro_option;
        micro_option.arena = arena;
        micro_option.arena_size = arena_size;
        micro_option.model = model;

        return &micro_option;
    }

    /****************************************************************************
     * Name: TfLiteMicroInterpreterOptionCreate
     *
     * Description:
     *    Create an interpreter option.
     *
     * Returned Value:
     *  If the model option is valid returns a pointer struct.
     *
     ****************************************************************************/
    TfLiteMicroInterpreterOption *TfLiteMicroInterpreterOptionCreate(void)
    {
        static TfLiteMicroInterpreterOption micro_interpreter_option;

        micro_interpreter_option.error_reporter = &micro_error_reporter;

        return &micro_interpreter_option;
    }

    /****************************************************************************
     * Name: TfLiteMicroInterpreterCreate
     *
     * Description:
     *   Create an interpreter pointer considering the current options.
     *
     * Parameters:
     *   interpreter_option : Pointer to interpeter option
     *   model_option       : Pointer to model option
     *
     * Returned Value:
     *  If the options are valid return a pointer struct otherwise nullptr.
     *
     ****************************************************************************/
    TfLiteMicroInterpreter *TfLiteMicroInterpreterCreate(TfLiteMicroInterpreterOption *interpreter_option, TfLiteMicroModelOption *model_option)
    {
        if (interpreter_option == nullptr || model_option == nullptr)
        {
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

    /****************************************************************************
     * Name: TfLiteMicroInterpreterDelete
     *
     * Description:
     *   Clear interpreter.
     *
     * Parameters:
     *   interpeter : Pointer to interpreter
     *
     ****************************************************************************/
    void TfLiteMicroInterpreterOptionDelete(TfLiteMicroInterpreterOption *option)
    {
        if (option != NULL)
        {
            memset(option, 0x00, sizeof(TfLiteMicroInterpreterOption));
        }
    }

    /****************************************************************************
     * Name: TfLiteMicroModelDelete
     *
     * Description:
     *   Clear the model.
     *
     * Parameters:
     *   model : Pointer to model.
     *
     ****************************************************************************/
    void TfLiteMicroModelDelete(TfLiteMicroModelOption *model)
    {
        if (model != NULL)
        {
            memset(model, 0x00, sizeof(TfLiteMicroModelOption));
        }
    }

    /****************************************************************************
     * Name: TfLiteMicroInterpreterOptionDelete
     *
     * Description:
     *   Clear the interpreter option.
     *
     * Parameters:
     *   option : Pointer to interpreter option.
     *
     ****************************************************************************/
    void TfLiteMicroInterpreterDelete(TfLiteMicroInterpreter *interpreter)
    {
        if (interpreter != NULL)
        {
            memset(interpreter, 0x00, sizeof(TfLiteMicroInterpreter));
        }
    }

    /****************************************************************************
     * Name: TfLiteMicroInterpreterAllocateTensors
     *
     * Description:
     *   Runs through the model and allocates all necessary input, output and intermediate Tensors
     *
     * Parameters:
     *   inter : Pointer to the interpreter.
     *
     * Returned Value:
     *   If allocated with success return kTfLiteOk otherwise kTfLiteError.
     *
     ****************************************************************************/
    TfLiteStatus TfLiteMicroInterpreterAllocateTensors(TfLiteMicroInterpreter *inter)
    {
        return inter->interpreter->AllocateTensors();
    }

    /****************************************************************************
     * Name: TfLiteMicroInterpreterGetInput
     *
     * Description:
     *   Get the input Tensor.
     *
     * Parameters:
     *   inter : Pointer to the interpreter.
     *   index : Index to the pointer.
     *
     * Returned Value:
     *  If the index pointer is out of length return nullptr otherwise return the Tensor pointer.
     *
     ****************************************************************************/
    TfLiteTensor *TfLiteMicroInterpreterGetInput(const TfLiteMicroInterpreter *inter, int32_t index)
    {
        return inter->interpreter->input(index);
    }

    /****************************************************************************
     * Name: TfLiteMicroInterpreterGetOutput
     *
     * Description:
     *   Get the output Tensor.
     *
     * Parameters:
     *   inter : Pointer to the interpreter.
     *   index : Index to the pointer.
     *
     * Returned Value:
     *  If the index pointer is out of length return nullptr otherwise return the Tensor pointer.
     *
     ****************************************************************************/
    TfLiteTensor *TfLiteMicroInterpreterGetOutput(const TfLiteMicroInterpreter *inter, int32_t index)
    {
        return inter->interpreter->output(index);
    }

    /****************************************************************************
     * Name: TfLiteMicroInterpreterGetOutputSize
     *
     * Description:
     *   Get the output buffer size.
     *
     * Parameters:
     *   inter : Pointer to the interpreter.
     *
     * Returned Value:
     *   Returns of the buffer size.
     *
     ****************************************************************************/
    int32_t TfLiteMicroInterpreterGetOutputSize(const TfLiteMicroInterpreter *inter)
    {
        return static_cast<int32_t>(inter->interpreter->outputs_size());
    }

    /****************************************************************************
     * Name: TfLiteMicroInterpreterGetInputSize
     *
     * Description:
     *   Get the intput buffer size.
     *
     * Parameters:
     *   inter : Pointet to the interpreter.
     *
     * Returned Value:
     *  Returns of the buffer size.
     *
     ****************************************************************************/
    int32_t TfLiteMicroInterpreterGetInputSize(const TfLiteMicroInterpreter *inter)
    {
        return static_cast<int32_t>(inter->interpreter->inputs_size());
    }

    /****************************************************************************
     * Name: TfLiteMicroInterpreterInvoke
     *
     * Description:
     *   Partial graph runs for stridden models.
     *
     * Parameters:
     *   inter : Pointer to the interpreter.
     *
     * Returned Value:
     *   Returns kTfLiteOk if can execute otherwise kTfLiteError.
     *
     ****************************************************************************/
    TfLiteStatus TfLiteMicroInterpreterInvoke(TfLiteMicroInterpreter *inter)
    {
        return inter->interpreter->Invoke();
    }

    /****************************************************************************
     * Name: TfLiteMicroTensorQuantizationParams
     *
     * Description:
     *   Get the quantization information.
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *
     * Returned Value:
     *   Returns the quantization information.
     *
     ****************************************************************************/
    TfLiteQuantizationParams TfLiteMicroTensorQuantizationParams(const TfLiteTensor *tensor)
    {
        return tensor->params;
    }

    /****************************************************************************
     * Name: TfLiteMicroGeInt8tData
     *
     * Description:
     *   Get the tensor data pointer int8 type.
     *
     * Parameters:
     *   tensor : Pointer Tensor.
     *   index  : Index to the pointer.
     *
     * Returned Value:
     *   Returns data pointer value.
     *
     ****************************************************************************/
    int8_t TfLiteMicroGeInt8tData(const TfLiteTensor *tensor, int index)
    {
        return tensor->data.int8[index];
    }

    /****************************************************************************
     * Name: TfLiteMicroSetInt8Data
     *
     * Description:
     *   Set the tensor data pointer int8 type.
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *   index  : Index to the pointer.
     *   value  : Value defined.
     *
     ****************************************************************************/
    void TfLiteMicroSetInt8Data(const TfLiteTensor *tensor, int index, int8_t value)
    {
        tensor->data.int8[index] = value;
    }

    /****************************************************************************
     * Name: TfLiteMicroSetData
     *
     * Description:
     *   Get the Tensor data pointer according to the type.
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *   index  : Index to the pointer.
     *   value  : Value defined.
     *   type   : Tensor type.
     *
     ****************************************************************************/
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
 
    /****************************************************************************
     * Name: TfLiteMicroGetData
     *
     * Description:
     *   Get the Tensor data pointer according to the type.
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *   index  : Index to the pointer.
     *   value  : Value defined.
     *   type   : Tensor type.
     *
     ****************************************************************************/
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
            return 0;
        }

    }

    /****************************************************************************
     * Name: TfLiteMicroGetFloatData
     *
     * Description:
     *   Get the Tensor data pointer float type.
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *   index  : Index to the pointer.
     *
     * Returned Value:
     *   Returns the data pointer value
     *
     ****************************************************************************/
    float TfLiteMicroGetFloatData(const TfLiteTensor *tensor, int index)
    {
        return tensor->data.f[index];
    }

     /****************************************************************************
     * Name: TfLiteMicroSetFloatData
     *
     * Description:
     *   Set Tensor data pointer float type.
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *   index  : Index to the pointer.
     *   value  : Value defined.
     *
     ****************************************************************************/
    void TfLiteMicroSetFloatData(const TfLiteTensor *tensor, int index, float value)
    {
        tensor->data.f[index] = value;
    }

    /****************************************************************************
     * Name: TfLiteMicroDimsData
     *
     * Description:
     *   Get the dimension data (input/output).
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *   index  : Index to the pointer.
     *
     * Returned Value:
     *   Returns the data value.
     *
     ****************************************************************************/
    int TfLiteMicroDimsData(const TfLiteTensor *tensor, int index)
    {
        return tensor->dims->data[index];
    }

    /****************************************************************************
     * Name: TfLiteMicroDimsSizeData
     *
     * Description:
     *   Get the dimension data size (input/output).
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *
     * Returned Value:
     *   Returns the data size.
     *
     ****************************************************************************/
    int TfLiteMicroDimsSizeData(const TfLiteTensor *tensor)
    {
        return tensor->dims->size;
    }

    /****************************************************************************
     * Name: TfLiteMicroGetType
     *
     * Description:
     *   Get the current Tensor type.
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *
     * Returned Value:
     *   Types supported by Tensor.
     *
     ****************************************************************************/
    TfLiteType TfLiteMicroGetType(const TfLiteTensor *tensor)
    {
        return tensor->type;
    }

    /****************************************************************************
     * Name: TfLiteMicroGetByte
     *
     * Description:
     *  Get the number of bytes required to store the data of the Tensor.
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *
     * Returned Value:
     *   Number of bytes.
     *
     ****************************************************************************/
    int TfLiteMicroGetByte(const TfLiteTensor *tensor)
    {
        return tensor->bytes;
    }
}