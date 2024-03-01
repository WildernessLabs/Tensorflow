/**
 * 
*/
#include "tensorflow/lite/core/api/c_api.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "tensorflow/lite/micro/spresense/debug_log_callback.h"
#include "c_api.h"

namespace
{

    // class CallbackMicroErrorReporter : public tflite::MicroErrorReporter{
    //     public:
    //         explicit CallbackMicroErrorReporter(TfLiteMicroErrorReporterCallback callback)
    //             :callback_(callback){}

    //     int Report(const char* format, va_list args)override {
    //         callback_.error_reporter(callback_.user_data, format, args);
    //         return 0;
    //     }

    //     private:
    //     TfLiteMicroErrorReporterCallback callback_;
    // };

}

extern "C"
{
    TfLiteMicroModel *TfLiteMicroGetModel(const void *model_data)
    {
        auto model = tflite::GetModel(model_data);

        if (model->version() != TFLITE_SCHEMA_VERSION)
        {
            /**
             * Show message error or return some status
            */
           return nullptr;
        }

        TfLiteMicroModel *tf_lite_micro_model = new TfLiteMicroModel;
        tf_lite_micro_model->model = std::shared_ptr<const tflite::Model>(model);

        return tf_lite_micro_model;
    }

    void TfLiteMicroModelDelete(TfLiteMicroModel *model)
    {
        delete model;
    }
    
    TfLiteMicroInterpreterOption *TfLiteMicroInterpreterOptionCreate()
    {
        return new TfLiteMicroInterpreterOption{};
    }

    void TfLiteMicroInterpreterOptionDelete(TfLiteMicroInterpreterOption *option)
    {
        delete option;
    }

    TfLiteMicroInterpreter *TfLiteMicroInterpreterCreate(const TfLiteMicroModel *impl, const TfLiteMicroInterpreterOption *_option)
    {
        if (!impl || impl->model)
        {
            return nullptr;
        }

        // std::unique_ptr<tflite::ErrorReporter> option_error_reporter;
        // if (option_error_reporter && _option->error_reporter_calback.error_reporter != nullptr)
        // {
        //    option_error_reporter = std::unique_ptr<CallbackMicroErrorReporter>(new CallbackMicroErrorReporter(_option->error_reporter_calback));
        // }
        tflite::MicroErrorReporter error;
        // TODO : Create a method for selecting the number of OpResolver (MicroMutableOpResolver)
        std::unique_ptr<tflite::MicroInterpreter> micro_interpreter(new tflite::MicroInterpreter(impl->model.get(), 
                                                                                tflite::AllOpsResolver{},
                                                                                _option->arena,
                                                                                _option->arena_size,
                                                                                &error,
                                                                                nullptr,
                                                                                nullptr));
        return new TfLiteMicroInterpreter{impl->model, std::move(micro_interpreter)};
    }

    void TfLiteMicroInterpreterDelete(TfLiteMicroInterpreter *option)
    {
        delete option;
    }

    TfLiteStatus TfLiteMicroInterpreterAllocateTensors(TfLiteMicroInterpreter *interpreter)
    {
        return interpreter->impl->AllocateTensors();
    }

    TfLiteTensor *TfLiteMicroInterpreterGetInput(const TfLiteMicroInterpreter *interpreter, int32_t input_index)
    {
        return interpreter->impl->input(input_index);
    }

    TfLiteTensor *TfLiteMicroInterpreterGetOutput(const TfLiteMicroInterpreter *interpreter, int32_t output_index)
    {
        return interpreter->impl->output(output_index);
    }

    int32_t TfLiteMicroInterpreterGetOutputCount(const TfLiteMicroInterpreter *interpreter)
    {
        return static_cast<int32_t>(interpreter->impl->outputs_size());
    }

    int32_t TfLiteMicroInterpreterGetInputCount(const TfLiteMicroInterpreter *interpreter)
    {
        return static_cast<int32_t>(interpreter->impl->inputs_size());
    }
    
    TfLiteStatus TfLiteMicroInterpreterInvoke(TfLiteMicroInterpreter *interpreter)
    {
        return interpreter->impl->Invoke();
    }

    TfLiteQuantizationParams TfLiteMicroTensorQuantizationParams (const TfLiteTensor *tensor)
    {
        return tensor->params;
    }
}