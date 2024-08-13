/****************************************************************************
 * c_api.h
 *
 *   Copyright (C) 2024 Wilderness Labs. All rights reserved.
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

    TFL_CAPI_EXPORT TfLiteMicroModelOption *TfLiteMicroGetModel(int arena_size, uint8_t *arena, const void *model_data);

    TFL_CAPI_EXPORT TfLiteMicroInterpreterOption *TfLiteMicroInterpreterOptionCreate(void);

    TFL_CAPI_EXPORT TfLiteMicroInterpreter *TfLiteMicroInterpreterCreate(TfLiteMicroInterpreterOption *interpreter_option, TfLiteMicroModelOption *model_option);

    TFL_CAPI_EXPORT void TfLiteMicroInterpreterDelete(TfLiteMicroInterpreter *interpreter);

    TFL_CAPI_EXPORT void TfLiteMicroModelDelete(TfLiteMicroModelOption *model);

    TFL_CAPI_EXPORT void TfLiteMicroInterpreterOptionDelete(TfLiteMicroInterpreterOption *option);

    TFL_CAPI_EXPORT TfLiteStatus TfLiteMicroInterpreterAllocateTensors(TfLiteMicroInterpreter *inter);

    TFL_CAPI_EXPORT TfLiteTensor *TfLiteMicroInterpreterGetInput(const TfLiteMicroInterpreter *inter, int32_t index);

    TFL_CAPI_EXPORT TfLiteTensor *TfLiteMicroInterpreterGetOutput(const TfLiteMicroInterpreter *inter, int32_t index);

    TFL_CAPI_EXPORT int8_t TfLiteMicroGeInt8tData(const TfLiteTensor *tensor, int index);

    TFL_CAPI_EXPORT void TfLiteMicroSetInt8Data(const TfLiteTensor *tensor, int index, int8_t value);

    TFL_CAPI_EXPORT int32_t TfLiteMicroInterpreterGetOutputSize(const TfLiteMicroInterpreter *inter);

    TFL_CAPI_EXPORT int32_t TfLiteMicroInterpreterGetInputSize(const TfLiteMicroInterpreter *inter);

    TFL_CAPI_EXPORT TfLiteStatus TfLiteMicroInterpreterInvoke(TfLiteMicroInterpreter *inter);

    TFL_CAPI_EXPORT TfLiteQuantizationParams TfLiteMicroTensorQuantizationParams(const TfLiteTensor *tensor);

    TFL_CAPI_EXPORT int TfLiteMicroDimsData(const TfLiteTensor *tensor, int index);

    TFL_CAPI_EXPORT int TfLiteMicroDimsSizeData(const TfLiteTensor *tensor);

    TFL_CAPI_EXPORT TfLiteType TfLiteMicroGetType(const TfLiteTensor *tensor);

    TFL_CAPI_EXPORT int TfLiteMicroGetByte(const TfLiteTensor *tensor);

    TFL_CAPI_EXPORT float TfLiteMicroGetFloatData(const TfLiteTensor *tensor, int index);

    TFL_CAPI_EXPORT void TfLiteMicroSetFloatData(const TfLiteTensor *tensor, int index, float value);

    TFL_CAPI_EXPORT void TfLiteMicroSetData(const TfLiteTensor *tensor, int index, void *value, TfLiteType type);

    TFL_CAPI_EXPORT void *TfLiteMicroGetData(const TfLiteTensor *tensor, int index, TfLiteType type);

#ifdef __cplusplus
}
#endif

#endif