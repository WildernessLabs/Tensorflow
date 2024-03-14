/****************************************************************************
 * c_api_internal.h
 *
 *   Copyright (C) 2020 Wilderness Labs. All rights reserved.
 *   Author: Wilderness Labs
 * 
 ****************************************************************************/

#ifndef TENSORFLOW_LITE_MICRO_C_C_API_INTERNAL_H_
#define TENSORFLOW_LITE_MICRO_C_C_API_INTERNAL_H_
#include <stdarg.h>

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/core/api/flatbuffer_conversions.h"
#include "tensorflow/lite/core/api/c_api.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

struct TfLiteMicroErrorReporterCallback
{

    void *user_data = nullptr;

    void (*error_reporter)(void *user_data, const char *format,
                           va_list args) = nullptr;
};

struct TfLiteMicroInterpreter
{
    tflite::MicroInterpreter *interpreter;
};

struct TfLiteMicroModelOption
{

    int arena_size;
    uint8_t *arena;
    const tflite::Model *model;
};

struct TfLiteMicroInterpreterOption
{
    tflite::ErrorReporter *error_reporter;
};

#endif