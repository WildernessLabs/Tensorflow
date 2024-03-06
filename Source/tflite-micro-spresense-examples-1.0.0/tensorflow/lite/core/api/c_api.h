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

    typedef struct TfLiteMicroModel TfLiteMicroModel;

    typedef struct TfLiteMicroInterpreterOption TfLiteMicroInterpreterOption;

    typedef struct TfLiteMicroModelOption TfLiteMicroModelOption;

    typedef struct TfLiteMicroInterpreter  TfLiteMicroInterpreter;

    TFL_CAPI_EXPORT TfLiteMicroModelOption *TfLiteMicroGetModel(uint8_t arena_size, uint8_t *arena, const void *model_data);

    TFL_CAPI_EXPORT void TfLiteMicroMalloc();

    TFL_CAPI_EXPORT TfLiteMicroInterpreter *TfLiteMicroInterpreterCreate(TfLiteMicroInterpreterOption *interpeter_option, TfLiteMicroModelOption *model_option);

    TFL_CAPI_EXPORT void TfLiteMicroInterpreterDelete(TfLiteMicroInterpreter *option);

    TFL_CAPI_EXPORT TfLiteStatus TfLiteMicroInterpreterAllocateTensors(TfLiteMicroInterpreter *inter);

    TFL_CAPI_EXPORT TfLiteTensor *TfLiteMicroInterpreterGetInput(const TfLiteMicroInterpreter *inter, int32_t input_index);

    TFL_CAPI_EXPORT TfLiteTensor *TfLiteMicroInterpreterGetOutput(const TfLiteMicroInterpreter *inter, int32_t output_index);

    TFL_CAPI_EXPORT int32_t TfLiteMicroInterpreterGetOutputCount(const TfLiteMicroInterpreter *inter);

    TFL_CAPI_EXPORT int32_t TfLiteMicroInterpreterGetInputCount(const TfLiteMicroInterpreter *inter);

    TFL_CAPI_EXPORT TfLiteStatus TfLiteMicroInterpreterInvoke(TfLiteMicroInterpreter *inter);

    TFL_CAPI_EXPORT TfLiteQuantizationParams TfLiteMicroTensorQuantizationParams (const TfLiteTensor *tensor);

#ifdef __cplusplus
}
#endif

#endif