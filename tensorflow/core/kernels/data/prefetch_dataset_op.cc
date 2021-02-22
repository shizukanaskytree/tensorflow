/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/kernels/data/prefetch_dataset_op.h"

#include <deque>
#include <thread>
#include <chrono>
#include <algorithm>    // std::next_permutation, std::sort

#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/model.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/dataset_utils.h"
#include "tensorflow/core/kernels/data/name_utils.h"
#include "tensorflow/core/kernels/data/stats_utils.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/stringprintf.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

namespace batch_util {
Status CopyElementToSlice(Tensor element, Tensor* parent, int64 index);
Status CopySliceToElement(const Tensor& parent, Tensor* element, int64 index);
Status MaybeMoveSliceToElement(Tensor* parent, Tensor* element, int64 index);
Status MoveSliceToElementSlice(Tensor* parent, Tensor* element, int64 p_index, int64 e_index);
Status CopyContiguousSlices(const Tensor& src, int64 src_offset,
                            int64 dst_offset, int64 num_slices, Tensor* dst);
}  // namespace batch_util

namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

/* static */ constexpr const char* const PrefetchDatasetOp::kDatasetType;
/* static */ constexpr const char* const PrefetchDatasetOp::kInputDataset;
/* static */ constexpr const char* const PrefetchDatasetOp::kBufferSize;
/* static */ constexpr const char* const PrefetchDatasetOp::kOutputTypes;
/* static */ constexpr const char* const PrefetchDatasetOp::kOutputShapes;
/* static */ constexpr const char* const PrefetchDatasetOp::kSlackPeriod;
/* static */ constexpr const char* const PrefetchDatasetOp::kLegacyAutotune;
/* static */ constexpr const char* const PrefetchDatasetOp::kBufferSizeMin;

namespace {

// Determines the fraction of slack time by which to delay prefetching of data.
constexpr double kSleepFactor = 0.2;
constexpr char kBuffer[] = "buffer";
constexpr char kStatus[] = "status";
constexpr char kSizeSuffix[] = ".size";
constexpr char kCodeSuffix[] = ".code";
constexpr char kErrorMessageSuffix[] = ".error_message";

}  // namespace

class PrefetchDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int64 buffer_size,
          int64 slack_period, bool legacy_autotune, int64 buffer_size_min)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        buffer_size_(buffer_size),
        slack_period_(slack_period),
        legacy_autotune_(legacy_autotune),
        buffer_size_min_(buffer_size_min) {
    input_->Ref();

    VLOG(0) << "TID: " << std::this_thread::get_id() << "; Dataset::Dataset" << ";\n"
            << ctx->op_kernel().requested_device();
    // 区分 GPU PrefetchDatasetOp and CPU

  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(Iterator::Params{
        this, name_utils::IteratorPrefix(kDatasetType, prefix)});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override {
    return name_utils::DatasetDebugString(kDatasetType);
  }

  int64 Cardinality() const override { return input_->Cardinality(); }

  Status InputDatasets(std::vector<const DatasetBase*>* inputs) const override {
    inputs->push_back(input_);
    return Status::OK();
  }

  Status CheckExternalState() const override {
    return input_->CheckExternalState();
  }

 protected:
  Status AsGraphDefInternal(SerializationContext* ctx,
                            DatasetGraphDefBuilder* b,
                            Node** output) const override {
    Node* input_graph_node = nullptr;
    TF_RETURN_IF_ERROR(b->AddInputDataset(ctx, input_, &input_graph_node));
    Node* buffer_size = nullptr;
    TF_RETURN_IF_ERROR(b->AddScalar(buffer_size_, &buffer_size));
    AttrValue slack_period_attr;
    b->BuildAttrValue(slack_period_, &slack_period_attr);
    AttrValue legacy_autotune_attr;
    b->BuildAttrValue(legacy_autotune_, &legacy_autotune_attr);
    AttrValue buffer_size_min_attr;
    b->BuildAttrValue(buffer_size_min_, &buffer_size_min_attr);

    TF_RETURN_IF_ERROR(
        b->AddDataset(this, {input_graph_node, buffer_size},
                      {std::make_pair(kSlackPeriod, slack_period_attr),
                       std::make_pair(kLegacyAutotune, legacy_autotune_attr),
                       std::make_pair(kBufferSizeMin, buffer_size_min_attr)},
                      output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          mu_(std::make_shared<mutex>()),
          cond_var_(std::make_shared<condition_variable>()),
          buffer_size_min_(params.dataset->buffer_size_min_),
          auto_tuner_(params.dataset->buffer_size_, buffer_size_min_),
          legacy_autotune_(params.dataset->legacy_autotune_),
          // If `legacy_autotune_`, initialize the `buffer_size_` value to be 0
          // to avoid the created node to be collected as tunable nodes in the
          // autotuning optimization.
          buffer_size_(std::make_shared<model::SharedState>(
              legacy_autotune_ ? 0 : params.dataset->buffer_size_, mu_,
              cond_var_)) {
      slack_us_ = 0;

      VLOG(0) << "TID: " << std::this_thread::get_id() << "; Iterator::Iterator";
    }

    ~Iterator() override {
      CancelThreads();
      if (deregister_fn_) deregister_fn_();
    }

    Status Initialize(IteratorContext* ctx) override {
      mutex_lock l(*mu_);
    
      // assume 2 tensors in the vector.
      //hide// reused_batched_elem_.value.reserve(2);

      // must specify data type and shape
      //hide// reused_batched_elem_.value.push_back(Tensor(DT_FLOAT, TensorShape({32,32,32,3})));
      //hide// reused_batched_elem_.value.push_back(Tensor(DT_INT32, TensorShape({32})));

      //hide// reused_batched_elem_.value = {Tensor(DT_FLOAT, TensorShape({128,32,32,3})), 
      //hide//                               Tensor(DT_INT32, TensorShape({128}))};

      //hide// // K reused_batched_elems_
      //hide// reused_batched_elems_.reserve(K_);
      //hide// for (int i = 0; i < K_; i++) {
      //hide//   reused_batched_elems_[i].value.reserve(2);
      //hide//   reused_batched_elems_[i].value.push_back(Tensor(DT_FLOAT, TensorShape({32,32,32,3})));
      //hide//   reused_batched_elems_[i].value.push_back(Tensor(DT_INT32, TensorShape({32})));
      //hide// }

      //VLOG(0) << "TID: " << std::this_thread::get_id() << "; Initiaize (Iterator)";
      
      if (buffer_size_->value == model::kAutotune) {
        buffer_size_->value = buffer_size_min_;
      }
      TF_RETURN_IF_ERROR(RegisterCancellationCallback(
          ctx->cancellation_manager(), [this]() { CancelThreads(); },
          &deregister_fn_));
      return dataset()->input_->MakeIterator(ctx, this, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      const auto& stats_aggregator = ctx->stats_aggregator();
      {
        mutex_lock l(*mu_);
        TF_RETURN_IF_ERROR(EnsurePrefetchThreadStarted(ctx));
        // Wait until the next element in the buffer has been
        // produced, or we are shutting down.
        if (legacy_autotune_) {
          while (!cancelled_ && buffer_.empty() && !prefetch_thread_finished_ &&
                 auto_tuner_.buffer_limit() != 0) {
            VLOG(0) << "empty";
            auto_tuner_.RecordEmpty();
            buffer_size_->value = auto_tuner_.buffer_limit();
            RecordStop(ctx);
            cond_var_->wait(l);
            RecordStart(ctx);
          }
        } else {
          while (!cancelled_ && buffer_.empty() && !prefetch_thread_finished_ &&
                 buffer_size_->value != 0) {
            VLOG(0) << "empty";
            RecordStop(ctx);
            cond_var_->wait(l);
            RecordStart(ctx);
          }
        }

        if (cancelled_) {
          return errors::Cancelled("Iterator was cancelled");
        }

        if (!buffer_.empty()) {
          return Consume(ctx, out_tensors, end_of_sequence);
        }

        if (prefetch_thread_finished_) {
          *end_of_sequence = true;
          return Status::OK();
        }

        DCHECK_EQ(buffer_limit(), 0);
      }

      mutex_lock input_l(input_mu_);
      {
        mutex_lock l(*mu_);
        if (stats_aggregator) {
          stats_aggregator->AddScalar(
              stats_utils::BufferSizeScalarName(dataset()->node_name()),
              static_cast<float>(buffer_.size()), num_elements());
          stats_aggregator->AddScalar(
              stats_utils::BufferCapacityScalarName(dataset()->node_name()),
              static_cast<float>(buffer_limit()), num_elements());
        }
        // Release mu_
      }
      return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeAsyncKnownRatioNode(
          std::move(args),
          /*ratio=*/1,
          {model::MakeParameter(kBufferSize, buffer_size_,
                                /*min=*/buffer_size_min_,
                                /*max=*/std::numeric_limits<int64>::max())});
    }

    Status SaveInternal(SerializationContext* ctx,
                        IteratorStateWriter* writer) override {
      // Acquire both locks to ensure that the prefetch thread and
      // all GetNext threads are blocked.
      mutex_lock input_l(input_mu_);
      mutex_lock l(*mu_);
      TF_RETURN_IF_ERROR(SaveInput(ctx, writer, input_impl_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(prefix(), kBufferSize, buffer_.size()));
      for (size_t i = 0; i < buffer_.size(); i++) {
        auto& buffer_element = buffer_[i];
        TF_RETURN_IF_ERROR(WriteStatus(writer, i, buffer_element.status));
        if (buffer_element.status.ok()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              absl::StrCat(prefix(), "::", i),
              absl::StrCat(kBuffer, kSizeSuffix), buffer_element.value.size()));
          for (size_t j = 0; j < buffer_element.value.size(); j++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                absl::StrCat(prefix(), "::", i),
                absl::StrCat(kBuffer, "[", j, "]"), buffer_element.value[j]));
          }
        }
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock input_l(input_mu_);
      mutex_lock l(*mu_);
      DCHECK(buffer_.empty());
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      size_t buffer_size;
      {
        int64 temp;
        TF_RETURN_IF_ERROR(reader->ReadScalar(prefix(), kBufferSize, &temp));
        buffer_size = static_cast<size_t>(temp);
      }
      for (size_t i = 0; i < buffer_size; i++) {
        buffer_.emplace_back();
        auto& buffer_element = buffer_.back();
        TF_RETURN_IF_ERROR(ReadStatus(reader, i, &buffer_element.status));
        if (buffer_element.status.ok()) {
          size_t value_size;
          {
            int64 temp;
            TF_RETURN_IF_ERROR(
                reader->ReadScalar(absl::StrCat(prefix(), "::", i),
                                   absl::StrCat(kBuffer, kSizeSuffix), &temp));
            value_size = static_cast<size_t>(temp);
          }
          buffer_element.value.reserve(value_size);
          for (size_t j = 0; j < value_size; j++) {
            buffer_element.value.emplace_back();
            TF_RETURN_IF_ERROR(
                reader->ReadTensor(absl::StrCat(prefix(), "::", i),
                                   absl::StrCat(kBuffer, "[", j, "]"),
                                   &buffer_element.value.back()));
          }
        }
        RecordBufferEnqueue(ctx, buffer_element.value);
      }
      return Status::OK();
    }

    data::TraceMeMetadata GetTraceMeMetadata() const override {
      int64 limit = -1, size = -1;
      data::TraceMeMetadata result;
      // NOTE: We only set the parallelism value if the lock can be acquired
      // right away to avoid introducing tracing overhead.
      if (mu_->try_lock()) {
        limit = buffer_limit();
        size = buffer_.size();
        if (!buffer_.empty()) {
          std::vector<std::string> shapes(buffer_.front().value.size());
          for (const auto& component : buffer_.front().value) {
            shapes.push_back(component.shape().DebugString());
          }
          result.push_back(std::make_pair("next_element_shapes",
                                          absl::StrJoin(shapes, ",")));
        }
        mu_->unlock();
      }
      result.push_back(std::make_pair(
          "buffer_limit",
          strings::Printf("%lld", static_cast<long long>(limit))));
      result.push_back(std::make_pair(
          "buffer_size",
          strings::Printf("%lld", static_cast<long long>(size))));
      result.push_back(std::make_pair(
          "autotune",
          dataset()->buffer_size_ == model::kAutotune ? "true" : "false"));
      result.push_back(std::make_pair(
          "autotune_mode", legacy_autotune_ ? "legacy" : "performance"));
      if (dataset()->slack_period_ > 0) {
        result.push_back(std::make_pair(
            "slack",
            strings::Printf("%lld", static_cast<long long>(slack_us_.load()))));
      }
      return result;
    }

   private:
    // A buffer element comprises a status and (if that status is
    // OK) a vector of tensors, representing an element of the input dataset.
    struct BufferElement {
      BufferElement() : uid(tensorflow::EnvTime::NowNanos()) {}

      // The producer sets `status` if getting the input element fails.
      Status status;
      // The buffered data element.
      std::vector<Tensor> value;
      int64 created_us;
      const uint64 uid;
    };

    int64 buffer_limit() const TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (legacy_autotune_) {
        return auto_tuner_.buffer_limit();
      }
      return buffer_size_->value;
    }

    void CancelThreads() TF_LOCKS_EXCLUDED(mu_) {
      mutex_lock l(*mu_);
      cancelled_ = true;
      cond_var_->notify_all();
    }

    Status Consume(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                   bool* end_of_sequence) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      const auto& stats_aggregator = ctx->stats_aggregator();
      if (stats_aggregator) {
        double buffer_limit_ = buffer_limit();
        stats_aggregator->AddToHistogram(
            stats_utils::BufferUtilizationHistogramName(dataset()->node_name()),
            {static_cast<float>(buffer_.size()) /
             static_cast<float>(buffer_limit_)},
            num_elements());
        stats_aggregator->AddScalar(
            stats_utils::BufferSizeScalarName(dataset()->node_name()),
            static_cast<float>(buffer_.size()), num_elements());
        stats_aggregator->AddScalar(
            stats_utils::BufferCapacityScalarName(dataset()->node_name()),
            static_cast<float>(buffer_limit_), num_elements());
      }
      // A new element is available. Forward the status from computing it, and
      // (if we successfully got an element) the output values.
      Status s = buffer_.front().status;
      if (s.ok()) {
        int64 buffer_element_id = buffer_.front().uid;
        profiler::TraceMe traceme(
            [&] {
              return profiler::TraceMeEncode(
                  "PrefetchConsume", {{"element_id", buffer_element_id}});
            },
            profiler::kInfo);
        if (dataset()->slack_period_ > 0 &&
            (num_elements() + 1) % dataset()->slack_period_ == 0) {
          // TODO(rachelim): Consider doing something more sophisticated
          // to decide how long to sleep for; e.g. using a kalman filter.
          int64 slack_us = EnvTime::NowMicros() - buffer_.front().created_us;
          // Every slack_period_-th element, update the most recent slack time,
          // measured by the duration between when the element is prefetched
          // and when it is consumed. We add kSleepFactor * slack_us_ to the
          // measurement because we slept for that duration before prefetching
          // the element.
          slack_us_ = kSleepFactor * slack_us_ + slack_us;
          VLOG(2) << "Setting slack_us_: " << slack_us_;
        }

        VLOG(0) << "TID: " << std::this_thread::get_id() << ", Consume buffer_: " << buffer_.size()
                << ", batch size: " << buffer_.front().value[0].dim_size(0);

        *out_tensors = std::move(buffer_.front().value);
        RecordBufferDequeue(ctx, *out_tensors);
      } else {
        // If status not ok, we still record the dequeue event to make sure each
        // enqueue event is paired with a dequeue event even in the presence of
        // errors.
        RecordBufferDequeue(ctx, buffer_.front().value);
      }
      if (legacy_autotune_) {
        auto_tuner_.RecordConsumption(buffer_.size());
        buffer_size_->value = auto_tuner_.buffer_limit();
      }
      buffer_.pop_front();
      *end_of_sequence = false;

      // Wake the prefetch thread, in case it has been waiting for space
      // in the buffer. Also wake up threads from other calls to GetNext.
      //
      // TODO(mrry): Consider using different condition variables for
      // GetNext and Prefetch.
      cond_var_->notify_all();
      return s;
    }

    Status EnsurePrefetchThreadStarted(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      if (!prefetch_thread_) {
        std::shared_ptr<IteratorContext> new_ctx =
            std::make_shared<IteratorContext>(*ctx);
        prefetch_thread_ = ctx->StartThread(
            "tf_data_prefetch", [this, new_ctx]() { PrefetchThread(new_ctx); });
      }
      return Status::OK();
    }

    // Prefetches elements of the input, storing results in an internal buffer.
    //
    // It owns the iterator context passed to it.
    void PrefetchThread(const std::shared_ptr<IteratorContext>& ctx) {
      //VLOG(0) << "PrefetchThread tid: " << std::this_thread::get_id();

      RecordStart(ctx.get());
      auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
      // Keep track of where we are in an iteration "burst"
      int num_produced = 0;
      while (true) {
        // 1. Wait for a slot in the buffer.
        {
          mutex_lock l(*mu_);
          while (!cancelled_ && buffer_.size() >= buffer_limit()) {
            VLOG(0) << ">= buffer limit: " << buffer_.size();
            RecordStop(ctx.get());
            cond_var_->wait(l);
            RecordStart(ctx.get());
            VLOG(0) << "exit buffer limit: " << buffer_.size();
          }

          if (cancelled_) {
            prefetch_thread_finished_ = true;
            cond_var_->notify_all();
            return;
          }
        }

        if (dataset()->slack_period_ > 0 &&
            num_produced % dataset()->slack_period_ == 0) {
          // For the first element in the "burst", sleep for a bit if there is
          // slack.
          VLOG(2) << "Sleeping for: " << slack_us_ * kSleepFactor;
          ctx->env()->SleepForMicroseconds(slack_us_ * kSleepFactor);
        }

        // 2. Read the next element.
        // Acquire the input mutex since we will be reading an element from the
        // input iterator. Note that we do not wish to release this mutex till
        // we have added the fetched element to the `buffer_` else there will be
        // local state that may be missed by SaveInternal.
        mutex_lock input_l(input_mu_);
        bool end_of_sequence;
        BufferElement buffer_element;
        {
          profiler::TraceMe traceme(
              [&] {
                return profiler::TraceMeEncode(
                    "PrefetchProduce", {{"element_id", buffer_element.uid}});
              },
              profiler::kInfo);
          buffer_element.status = input_impl_->GetNext(
              ctx.get(), &buffer_element.value, &end_of_sequence);
        }
        if (buffer_element.status.ok() && end_of_sequence) {
          mutex_lock l(*mu_);
          prefetch_thread_finished_ = true;
          cond_var_->notify_all();
          return;
        }

        // 3. Signal that the element has been produced.
        {
          mutex_lock l(*mu_);
          RecordBufferEnqueue(ctx.get(), buffer_element.value);
          buffer_element.created_us = EnvTime::NowMicros();
          buffer_.push_back(buffer_element);

          // wxf
          //hide// VLOG(0) << "*** vector of tensors size: " << buffer_element.value.size();
          //hide// for (auto& it: buffer_element.value) {
          //hide//   VLOG(0) << "*** tensor debug: " << it.DebugString();
          //hide// }
          //hide// *** vector of tensors size: 2
          //hide// *** tensor debug: Tensor<type: float shape: [32,32,32,3] values: [[[-1.28277624 -1.28277624 -1.28277624]]]...>
          //hide// *** tensor debug: Tensor<type: int32 shape: [32] values: 2 9 8...>
          //~wxf

          // 6 threads, 
          // - 2 CPU tensors, batch_size = 128
          // - 2 CPU tensors, batch_size = 128/num_gpus = 32 or 64 
          // - 2 GPU tensors, batch_size = 128/num_gpus = 32 or 64 
          // so, the c++ code should deal with python code

          int batch_size = buffer_element.value[0].dim_size(0);
          //hide// VLOG(0) << "BATCH SIZE: " << batch_size;
          //hide// if (batch_size == 64) {
          //hide//   VLOG(0) << buffer_element.value[0].DebugString();
          //hide//   VLOG(0) << buffer_element.value[1].DebugString();
          //hide//   std::this_thread::sleep_for(std::chrono::milliseconds(4000));
          //hide// } else {
          //hide//   VLOG(0) << buffer_element.value[0].DeviceSafeDebugString();
          //hide//   VLOG(0) << buffer_element.value[1].DeviceSafeDebugString();
          //hide//   std::this_thread::sleep_for(std::chrono::milliseconds(4000));
          //hide// }

          // batch_size: 128 and 32 
          // total batch size is in the CPU size.
          
          //hide// if (batch_size == cached_buffer_pool_size_) { // selected
          if (false) { // disable it.

            if (cached_buffer_.size() < batch_size) {
              cached_buffer_.push_back(buffer_element);
              VLOG(0) << "TID: " << std::this_thread::get_id() << "; \n"
                      << "buffer_element size: " << buffer_element.value.size() << "; \n" 
                      << "[0]" << buffer_element.value[0].DeviceSafeDebugString() << "; \n" 
                      << "[1]" << buffer_element.value[1].DeviceSafeDebugString();
              // how to set prefetch thread in dataset tensorflow?
            } else {

              // random permutation
              std::vector<int> selected(batch_size);
              for (int i = 0; i < batch_size; i++) {
                selected[i] = i;
              }

              // init 
              reused_batched_elem_.value.reserve(2);

              std::vector<Tensor>& new_tensors = reused_batched_elem_.value;
              Tensor& batched_images = new_tensors[0];
              batched_images.CopyFrom(buffer_element.value[0], buffer_element.value[0].shape());
              //VLOG(0) << "batched_images size: " << batched_images.DebugString();

              Tensor& batched_labels = new_tensors[1];
              batched_labels.CopyFrom(buffer_element.value[1], buffer_element.value[1].shape());
              //VLOG(0) << "batched_labels size: " << batched_labels.DebugString();

              // for loop of K START here...
              //hide// for (int k = 0; k < K_; k++) {

                //VLOG(0) << "TID: " << std::this_thread::get_id() << "; k: " << k;

                //VLOG(0) << "permutation: ";
                std::random_shuffle(selected.begin(), selected.end());
                //hide// string output;
                //hide// for (int i = 0; i < batch_size; i++) {
                //hide//   output += std::to_string(selected[i]) + "\t";
                //hide// }
                //hide// VLOG(0) << output;
                //hide// std::this_thread::sleep_for(std::chrono::milliseconds(2000));

                // iterate all cached_buffer_
                int i = 0;
                for (auto& tensors: cached_buffer_) {

                  bool is_images = true;
                  for (auto& tensor: tensors.value) {
                    if (is_images) {
                      batch_util::MoveSliceToElementSlice(&tensor, &batched_images, selected[i], i);
                      //VLOG(0) << "TID: " << std::this_thread::get_id() << "; i: " << i << " ;\n"
                      //        << "origin: " << tensor.DebugString(5) << " ;\n"
                      //        << "images: " << batched_images.DebugString(5);
                    } else {
                      batch_util::MoveSliceToElementSlice(&tensor, &batched_labels, selected[i], i);
                      //VLOG(0) << "TID: " << std::this_thread::get_id() << "; i: " << i << " ;\n"
                      //        << "origin: " << tensor.DebugString(10) << " ;\n"
                      //        << "labels: " << batched_labels.DebugString(10);
                    }

                    //hide// std::this_thread::sleep_for(std::chrono::milliseconds(30));
                    is_images = false;
                  }
                  VLOG(0) << "TID: " << std::this_thread::get_id() << "; i: " << i;
                  i++;
                  // check the result.
                  //hide// for (auto& tensor: tensors.value) {
                  //hide//   VLOG(0) << tensor.DebugString(32);
                  //hide// }
                }
                VLOG(0) << "TID: " << std::this_thread::get_id() << " inner K_ for end";
                
                buffer_.push_back(reused_batched_elem_);
                VLOG(0) << "TID: " << std::this_thread::get_id() << " push_back end";

                //hide// std::this_thread::sleep_for(std::chrono::milliseconds(2000));
                //hide// reused_batched_elems_.push_back(reused_batched_elem_);
              //hide// }
              // for loop of K END here...

              // replace a new buffer_element 
              //HIDE// cached_buffer_.pop_front();
              //HIDE// VLOG(0) << "TID: " << std::this_thread::get_id() << " pop_front end";
              //HIDE// cached_buffer_.push_back(buffer_element);
              //HIDE// VLOG(0) << "TID: " << std::this_thread::get_id() << " cached_buffer_ push_back end";
            }
          }

          //hide-baseline// // test google echo baseline        
          //hide-baseline// if (batch_size == cached_buffer_pool_size_) {          
          //hide-baseline//   for (int i = 0; i < K_; i++) {
          //hide-baseline//     buffer_.push_back(buffer_element);
          //hide-baseline//   }
          //hide-baseline// }
          
          //VLOG(0) << "prefetch buffer_ size: " << buffer_.size();

          //hide// buffer_.push_back(std::move(buffer_element));
          cond_var_->notify_all();
        }
        ++num_produced;
        //hide// num_produced += 1 + K_;
      }
    }

    Status WriteStatus(IteratorStateWriter* writer, size_t index,
                       const Status& status) TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(absl::StrCat(prefix(), "::", index), CodeKey(),
                              static_cast<int64>(status.code())));
      if (!status.ok()) {
        TF_RETURN_IF_ERROR(
            writer->WriteScalar(absl::StrCat(prefix(), "::", index),
                                ErrorMessageKey(), status.error_message()));
      }
      return Status::OK();
    }

    Status ReadStatus(IteratorStateReader* reader, size_t index, Status* status)
        TF_EXCLUSIVE_LOCKS_REQUIRED(*mu_) {
      int64 code_int;
      TF_RETURN_IF_ERROR(reader->ReadScalar(absl::StrCat(prefix(), "::", index),
                                            CodeKey(), &code_int));
      error::Code code = static_cast<error::Code>(code_int);

      if (code != error::Code::OK) {
        tstring error_message;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(absl::StrCat(prefix(), "::", index),
                               ErrorMessageKey(), &error_message));
        *status = Status(code, error_message);
      } else {
        *status = Status::OK();
      }
      return Status::OK();
    }

    string CodeKey() { return absl::StrCat(kStatus, kCodeSuffix); }

    string ErrorMessageKey() {
      return absl::StrCat(kStatus, kErrorMessageSuffix);
    }

    // This mutex is used to ensure exclusivity between multiple threads
    // reading/writing this iterator's local state.
    //
    // NOTE: We should never call GetNext on the input while holding this mutex.
    const std::shared_ptr<mutex> mu_;
    // This mutex is used to ensure exclusivity between multiple threads
    // accessing the input iterator. We keep this separate from `mu_` to allow
    // prefetching to run in parallel with GetNext calls.
    mutex input_mu_ TF_ACQUIRED_BEFORE(*mu_);
    std::unique_ptr<IteratorBase> input_impl_ TF_GUARDED_BY(input_mu_);
    const std::shared_ptr<condition_variable> cond_var_;
    const int64 buffer_size_min_;
    PrefetchAutotuner auto_tuner_ TF_GUARDED_BY(*mu_);
    std::deque<BufferElement> buffer_ TF_GUARDED_BY(*mu_);
    std::unique_ptr<Thread> prefetch_thread_ TF_GUARDED_BY(*mu_);
    bool cancelled_ TF_GUARDED_BY(*mu_) = false;
    bool prefetch_thread_finished_ TF_GUARDED_BY(*mu_) = false;
    const bool legacy_autotune_;

    std::atomic<int64> slack_us_;

    // If legacy_autotune_ is false, identifies the maximum size of the buffer.
    const std::shared_ptr<model::SharedState> buffer_size_;

    // Method for deregistering the cancellation callback.
    std::function<void()> deregister_fn_;
   public: 
    // for reuse
    // echo(repeat) times
    int K_ = 2;

    int cached_buffer_pool_size_ = 128;
    // it seems like it is thread-local. different thread has its own cached_buffer_.
    std::deque<BufferElement> cached_buffer_ TF_GUARDED_BY(*mu_);
    // the generated new batch from history cached buffer pool
    BufferElement reused_batched_elem_ TF_GUARDED_BY(*mu_);
    // a list of the generated new batch from history cached buffer pool
    std::vector<BufferElement> reused_batched_elems_ TF_GUARDED_BY(*mu_);
  };
  const DatasetBase* const input_;
  const int64 buffer_size_;

  // If non-zero, determines the period between injecting "slack" into the
  // execution.
  const int64 slack_period_;

  // Determines whether legacy autotuning should be used.
  const bool legacy_autotune_ = true;

  // If autotune is enabled, determines the minimal value of `buffer_size`
  // parameter.
  const int64 buffer_size_min_ = 0;

  TraceMeMetadata traceme_metadata_;
};

PrefetchDatasetOp::PrefetchDatasetOp(OpKernelConstruction* ctx)
    : UnaryDatasetOpKernel(ctx) {
  if (ctx->HasAttr(kSlackPeriod)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kSlackPeriod, &slack_period_));
  }
  if (ctx->HasAttr(kLegacyAutotune)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kLegacyAutotune, &legacy_autotune_));
  }
  if (ctx->HasAttr(kBufferSizeMin)) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kBufferSizeMin, &buffer_size_min_));
  }
}

void PrefetchDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {
  int64 buffer_size = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64>(ctx, kBufferSize, &buffer_size));
  OP_REQUIRES(ctx, buffer_size >= 0 || buffer_size == model::kAutotune,
              errors::InvalidArgument("buffer_size must be >= 0 or set "
                                      "buffer_size to be ",
                                      model::kAutotune, " for auto-tuning"));

  if (buffer_size == model::kAutotune) {
    metrics::RecordTFDataAutotune(kDatasetType);
  }

  *output = new Dataset(ctx, input, buffer_size, slack_period_,
                        legacy_autotune_, buffer_size_min_);
}

namespace {
REGISTER_KERNEL_BUILDER(Name("PrefetchDataset").Device(DEVICE_CPU).Priority(2),
                        PrefetchDatasetOp);
REGISTER_KERNEL_BUILDER(Name("PrefetchDataset")
                            .Device(DEVICE_GPU)
                            .HostMemory("buffer_size")
                            .HostMemory("input_dataset")
                            .HostMemory("handle")
                            .Priority(1),
                        PrefetchDatasetOp);
}  // namespace

}  // namespace data
}  // namespace tensorflow
