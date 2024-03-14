/****************************************************************************
 * c_api.h
 *
 *   Copyright (C) 2020 Wilderness Labs. All rights reserved.
 *   Author: Wilderness Labs
 * 
 ****************************************************************************/

#ifndef TENSORFLOW_LITE_MICRO_CORE_C_API_H_
#define TENSORFLOW_LITE_MICRO_CORE_C_API_H_

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "tensorflow/lite/c/c_api_internal.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct TfLiteTensor TfLiteMicroSensor;

    typedef struct TfLiteMicroInterpreterOption TfLiteMicroInterpreterOption;

    typedef struct TfLiteMicroModelOption TfLiteMicroModelOption;

    typedef struct TfLiteMicroInterpreter TfLiteMicroInterpreter;

    /****************************************************************************
     * Name: TfLiteMicroGetModel
     *
     * Description:
     *    Load the model store as a flat buffer and hold the arena buffer.
     *
     * Parameters:
     *   arena_size : Size of the arena
     *   arena      : Pointer to the arena
     *   model_data : Pointer to hold the model
     *
     * Returned Value:
     *  If the model corresponds to the current version of TF returns a pointer struct otherwise nullptr.
     *
     ****************************************************************************/

    TFL_CAPI_EXPORT TfLiteMicroModelOption *TfLiteMicroGetModel(int arena_size, uint8_t *arena, const void *model_data);

    /****************************************************************************
     * Name: TfLiteMicroInterpreterOptionCreate
     *
     * Description:
     *    .
     *
     * Parameters:
     *   option : Pointer to the model option
     *
     * Returned Value:
     *  If the model option is valid returns a pointer struct otherwise nullptr.
     *
     ****************************************************************************/

    TFL_CAPI_EXPORT TfLiteMicroInterpreterOption *TfLiteMicroInterpreterOptionCreate(TfLiteMicroModelOption *option);

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

    TFL_CAPI_EXPORT TfLiteMicroInterpreter *TfLiteMicroInterpreterCreate(TfLiteMicroInterpreterOption *interpreter_option, TfLiteMicroModelOption *model_option);

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

    TFL_CAPI_EXPORT void TfLiteMicroInterpreterDelete(TfLiteMicroInterpreter *interpreter);

    /****************************************************************************
     * Name: TfLiteMicroModelDelete
     *
     * Description:
     *   Clear the model.
     *
     * Parameters:
     *   model : Pointer to model
     *
     ****************************************************************************/

    TFL_CAPI_EXPORT void TfLiteMicroModelDelete(TfLiteMicroModelOption *model);

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

    TFL_CAPI_EXPORT void TfLiteMicroInterpreterOptionDelete(TfLiteMicroInterpreterOption *option);

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

    TFL_CAPI_EXPORT TfLiteStatus TfLiteMicroInterpreterAllocateTensors(TfLiteMicroInterpreter *inter);

    /****************************************************************************
     * Name: TfLiteMicroInterpreterGetInput
     *
     * Description:
     *   Get input Tensor.
     *
     * Parameters:
     *   inter : Pointer to the interpreter.
     *   index : Pointer index.
     *
     * Returned Value:
     *  If the index pointer is out of length return nullptr otherwise return the Tensor pointer.
     *
     ****************************************************************************/
    TFL_CAPI_EXPORT TfLiteTensor *TfLiteMicroInterpreterGetInput(const TfLiteMicroInterpreter *inter, int32_t index);

    /****************************************************************************
     * Name: TfLiteMicroInterpreterGetOutput
     *
     * Description:
     *   Get output Tensor.
     *
     * Parameters:
     *   inter : Pointer to the interpreter.
     *   index : Pointer index.
     *
     * Returned Value:
     *  If the index pointer is out of length return nullptr otherwise return the Tensor pointer.
     *
     ****************************************************************************/

    TFL_CAPI_EXPORT TfLiteTensor *TfLiteMicroInterpreterGetOutput(const TfLiteMicroInterpreter *inter, int32_t index);

    /****************************************************************************
     * Name: TfLiteMicroGeInt8tData
     *
     * Description:
     *   Get tensor data pointer int8 type.
     *
     * Parameters:
     *   tensor : Pointer Tensor.
     *   index  : Pointer index.
     *
     * Returned Value:
     *   Returns data pointer value.
     *
     ****************************************************************************/

    TFL_CAPI_EXPORT int8_t TfLiteMicroGeInt8tData(const TfLiteTensor *tensor, int index);

    /****************************************************************************
     * Name: TfLiteMicroSetInt8Data
     *
     * Description:
     *   Set tensor data pointer int8 type.
     *
     * Parameters:
     *   tensor : Pointer to the Tensor
     *   index  : Pointer index
     *   value  : Value defined
     *
     ****************************************************************************/

    TFL_CAPI_EXPORT void TfLiteMicroSetInt8Data(const TfLiteTensor *tensor, int index, int8_t value);

    /****************************************************************************
     * Name: TfLiteMicroInterpreterGetOutputSize
     *
     * Description:
     *   Get output buffer size.
     *
     * Parameters:
     *   inter : Pointer to the interpreter
     *
     * Returned Value:
     *   Returns of the buffer size.
     *
     ****************************************************************************/

    TFL_CAPI_EXPORT int32_t TfLiteMicroInterpreterGetOutputSize(const TfLiteMicroInterpreter *inter);

    /****************************************************************************
     * Name: TfLiteMicroInterpreterGetInputSize
     *
     * Description:
     *   Get intput buffer size.
     *
     * Parameters:
     *   inter : Pointet to the interpreter.
     *
     * Returned Value:
     *  Returns of the buffer size.
     *
     ****************************************************************************/

    TFL_CAPI_EXPORT int32_t TfLiteMicroInterpreterGetInputSize(const TfLiteMicroInterpreter *inter);

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

    TFL_CAPI_EXPORT TfLiteStatus TfLiteMicroInterpreterInvoke(TfLiteMicroInterpreter *inter);

    /****************************************************************************
     * Name: TfLiteMicroTensorQuantizationParams
     *
     * Description:
     *   Get quantization information.
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *
     * Returned Value:
     *   Returns the quantization information.
     *
     ****************************************************************************/

    TFL_CAPI_EXPORT TfLiteQuantizationParams TfLiteMicroTensorQuantizationParams(const TfLiteTensor *tensor);

    /****************************************************************************
     * Name: TfLiteMicroDimsData
     *
     * Description:
     *   Get the dimension data (input/output).
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *   index  : Pointer index.
     *
     * Returned Value:
     *   Returns the data value.
     *
     ****************************************************************************/

    TFL_CAPI_EXPORT int TfLiteMicroDimsData(const TfLiteTensor *tensor, int index);

    /****************************************************************************
     * Name: TfLiteMicroDimsSizeData
     *
     * Description:
     *   Get dimension data size (input/output).
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *
     * Returned Value:
     *   Returns the data size.
     *
     ****************************************************************************/

    TFL_CAPI_EXPORT int TfLiteMicroDimsSizeData(const TfLiteTensor *tensor);

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

    TFL_CAPI_EXPORT TfLiteType TfLiteMicroGetType(const TfLiteTensor *tensor);

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

    TFL_CAPI_EXPORT int TfLiteMicroGetByte(const TfLiteTensor *tensor);

    /****************************************************************************
     * Name: TfLiteMicroGetFloatData
     *
     * Description:
     *   Get Tensor data pointer float type.
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *   index  : Pointer index.
     *
     * Returned Value:
     *   Returns data pointer value
     *
     ****************************************************************************/

    TFL_CAPI_EXPORT float TfLiteMicroGetFloatData(const TfLiteTensor *tensor, int index);

    /****************************************************************************
     * Name: TfLiteMicroSetFloatData
     *
     * Description:
     *   Set Tensor data pointer float type.
     *
     * Parameters:
     *   tensor : Pointer to the Tensor.
     *   index  : Pointer index.
     *   value  : Value defined.
     *
     ****************************************************************************/

    TFL_CAPI_EXPORT void TfLiteMicroSetFloatData(const TfLiteTensor *tensor, int index, float value);

#ifdef __cplusplus
}
#endif

#endif