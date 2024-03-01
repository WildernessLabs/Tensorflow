#ifndef TENSORFLOW_LITE_MICRO_CORE_C_API_H_
#define TENSORFLOW_LITE_MICRO_CORE_C_API_H_

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "tensorflow/lite/c/c_api_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct TfLiteTensor TfLiteMicroSensor;

typedef struct TfLiteMicroModel TfLiteMicroModel;

typedef struct TfLiteMicroInterpreterOption TfLiteMicroInterpreterOption;

TFL_CAPI_EXPORT void TfLitMicroGetModel (const void *model_data);

TFL_CAPI_EXPORT TfLiteMicroInterpreterOption* TfLiteMicroInterpreterOptionCreate();

TFL_CAPI_EXPORT void TfLiteMicroModelDelete(TfLiteMicroModel *model);

TFL_CAPI_EXPORT void TfLiteMicroInterpreterOptionDelete(TfLiteMicroInterpreterOption* option);

TFL_CAPI_EXPORT TfLiteMicroInterpreter *TfLiteMicroInterpreterCreate(const TfLiteMicroModel *impl, const TfLiteMicroInterpreterOption *_option);

TFL_CAPI_EXPORT void TfLiteMicroInterpreterDelete(TfLiteMicroInterpreter* option);

TFL_CAPI_EXPORT TfLiteStatus TfLiteMicroInterpreterAllocateTensors(TfLiteMicroInterpreter* interpreter);

TFL_CAPI_EXPORT TfLiteTensor *TfTFL_CAPI_EXPORTLiteMicroInterpreterGetInput(const TfLiteMicroInterpreter* interpreter, int32_t input_index);

TFL_CAPI_EXPORT TfLiteTensor *TfLiteMicroInterpreterGetOutput(const TfLiteMicroInterpreter *interpreter, int32_t output_index);

TFL_CAPI_EXPORT int32_t TfLiteMicroInterpreterGetOutputCount(const TfLiteMicroInterpreter *interpreter);

TFL_CAPI_EXPORT int32_t TfLiteMicroInterpreterGetInputCount(const TfLiteMicroInterpreter *interpreter);

TFL_CAPI_EXPORT TfLiteStatus TfLiteMicroInterpreterInvoke(TfLiteMicroInterpreter *interpreter);

// TFL_CAPI_EXPORT TfLiteQuantizationParams TfLiteMicroTensorQuantizationParams (const TfLiteTensor *tensor);

#ifdef __cplusplus
    }
#endif

#endif