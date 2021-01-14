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
#include <chrono>
using namespace std::chrono;

#include "tensorflow/core/common_runtime/metrics.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/framework/stats_aggregator.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/data/stats_utils.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"

namespace tensorflow {
namespace data {

// See documentation in ../../ops/dataset_ops.cc for a high-level
// description of the following op.

// Determines the fraction of slack time by which to delay prefetching of data.
constexpr double kSleepFactor = 0.2;
constexpr char kDatasetName[] = "Prefetch";

// NOTE: find //debug// VLOG(0) to find buffer size info.

// 每隔多少步进行采样老数据, 然后缓存下来. 50,000 data for cifar-10, 使用质数来 sample.
//old// int stale_data_sampling_freq = 400; // good setting: 400.

// every 16 new mini-batches data, then we append some (K=16) history cached data to it.
//o// int echo_freq = 1; //16*6;

class PrefetchDatasetOp::Dataset : public DatasetBase {
 public:
  Dataset(OpKernelContext* ctx, const DatasetBase* input, int64 buffer_size,
          int64 slack_period)
      : DatasetBase(DatasetContext(ctx)),
        input_(input),
        buffer_size_(buffer_size),
        slack_period_(slack_period) {
    //VLOG(0) << "slack_period_: " << slack_period_; 
    //VLOG(0) << "buffer_size_: " << buffer_size_; 
    // buffer_size_: -1
    input_->Ref();
  }

  ~Dataset() override { input_->Unref(); }

  std::unique_ptr<IteratorBase> MakeIteratorInternal(
      const string& prefix) const override {
    return absl::make_unique<Iterator>(
        Iterator::Params{this, strings::StrCat(prefix, "::", kDatasetName)});
  }

  const DataTypeVector& output_dtypes() const override {
    return input_->output_dtypes();
  }

  const std::vector<PartialTensorShape>& output_shapes() const override {
    return input_->output_shapes();
  }

  string DebugString() const override { return "PrefetchDatasetOp::Dataset"; }

  int64 Cardinality() const override { return input_->Cardinality(); }

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
    TF_RETURN_IF_ERROR(b->AddDataset(
        this, {input_graph_node, buffer_size},
        {std::make_pair("slack_period", slack_period_attr)}, output));
    return Status::OK();
  }

 private:
  class Iterator : public DatasetIterator<Dataset> {
   public:
    explicit Iterator(const Params& params)
        : DatasetIterator<Dataset>(params),
          auto_tuner_(params.dataset->buffer_size_) {
      slack_us_ = 0;
    }

    ~Iterator() override {
      // Signal the prefetch thread to terminate it. We will then
      // join that thread when we delete `this->prefetch_thread_`.
      //
      // TODO(mrry): Replace this cancellation logic with a
      // CancellationManager. The syntax would be more heavyweight,
      // but it would be possible to thread a cancellation manager
      // through the IteratorContext to upstream,
      // potentially-blocking iterators, when we add these.
      {
        mutex_lock l(mu_);
        cancelled_ = true;
        cond_var_.notify_all();
      }
    }

    string BuildTraceMeName() override {
      int64 buffer_limit;
      {
        tf_shared_lock l(mu_);
        buffer_limit = auto_tuner_.buffer_limit();
        //VLOG(0) << "BuildTraceMeName: buffer_limit" << buffer_limit; 
      }
      return strings::StrCat(prefix(), "#buffer_limit=", buffer_limit, "#");
    }

    Status Initialize(IteratorContext* ctx) override {
      return dataset()->input_->MakeIterator(ctx, prefix(), &input_impl_);
    }

    Status GetNextInternal(IteratorContext* ctx,
                           std::vector<Tensor>* out_tensors,
                           bool* end_of_sequence) override {
      //VLOG(0) << "prefetch_dataset_op::GetNextInternal";
      const auto& stats_aggregator = ctx->stats_aggregator();
      {
        mutex_lock l(mu_);
        TF_RETURN_IF_ERROR(EnsurePrefetchThreadStarted(ctx));
        // Wait until the next element in the buffer has been
        // produced, or we are shutting down.
        //VLOG(0) << "- Before echoing, buffer size: " << buffer_.size(); 

        // 当 model 从 dataset pipeline 去取的时候, 如果是空的, buffer_.empty() 那么就用老数据.
        // 前提是 echoing_buffer_ 内有 elements!
        while (!echoing_buffer_.empty() && !cancelled_ && buffer_.empty() && 
               !prefetch_thread_finished_ && auto_tuner_.buffer_limit() != 0) {
          //VLOG(0) << "== Thread " << std::this_thread::get_id() << ", buffer size:" << buffer_.size();
          // when buffer_ is empty, push stale data
          for(int i = 0; i < K_; ++i){
            int r = rand() % (echoing_buffer_.size());
            auto elem = echoing_buffer_[r];
            buffer_.push_back(elem);
          }
          //VLOG(0) << "elem created time: " << elem.created_us;
          //VLOG(0) << "+ push to buffer, size: " << buffer_.size();;
        }
        //VLOG(0) << "+ After echoing, buffer size: "  

        while (!cancelled_ && buffer_.empty() && !prefetch_thread_finished_ &&
               auto_tuner_.buffer_limit() != 0) {
          //VLOG(0) << "GetNextInternal, wait, empty";      
          auto_tuner_.RecordEmpty();
          RecordStop(ctx);
          cond_var_.wait(l);
          RecordStart(ctx);
        }

        if (cancelled_) {
          return errors::Cancelled(
              "PrefetchDatasetOp::Dataset::Iterator::GetNext");
        }

        if (!buffer_.empty()) {
          //VLOG(0) << "Consume @ GetNextInternal since buffer not empty";
          return Consume(ctx, out_tensors, end_of_sequence);
        }

        if (prefetch_thread_finished_) {
          *end_of_sequence = true;
          return Status::OK();
        }

        DCHECK_EQ(auto_tuner_.buffer_limit(), 0);
      }

      mutex_lock parent_l(parent_mu_);
      mutex_lock l(mu_);
      if (stats_aggregator) {
        stats_aggregator->AddScalar(
            stats_utils::BufferSizeScalarName(dataset()->node_name()),
            static_cast<float>(buffer_.size()), num_elements());
        stats_aggregator->AddScalar(
            stats_utils::BufferCapacityScalarName(dataset()->node_name()),
            static_cast<float>(auto_tuner_.buffer_limit()), num_elements());
      }
      return input_impl_->GetNext(ctx, out_tensors, end_of_sequence);
    }

   protected:
    std::shared_ptr<model::Node> CreateNode(
        IteratorContext* ctx, model::Node::Args args) const override {
      return model::MakeAsyncKnownRatioNode(std::move(args),
                                            /*ratio=*/1,
                                            /*parameters=*/{});
    }

    Status SaveInternal(IteratorStateWriter* writer) override {
      // Acquire both locks to ensure that the prefetch thread and
      // all GetNext threads are blocked.
      mutex_lock parent_l(parent_mu_);
      mutex_lock l(mu_);
      TF_RETURN_IF_ERROR(SaveInput(writer, input_impl_));
      TF_RETURN_IF_ERROR(
          writer->WriteScalar(full_name("buffer_size"), buffer_.size()));
      for (size_t i = 0; i < buffer_.size(); i++) {
        auto& buffer_element = buffer_[i];
        TF_RETURN_IF_ERROR(WriteStatus(writer, i, buffer_element.status));
        if (buffer_element.status.ok()) {
          TF_RETURN_IF_ERROR(writer->WriteScalar(
              full_name(strings::StrCat("buffer[", i, "].size")),
              buffer_element.value.size()));
          for (size_t j = 0; j < buffer_element.value.size(); j++) {
            TF_RETURN_IF_ERROR(writer->WriteTensor(
                full_name(strings::StrCat("buffer[", i, "][", j, "]")),
                buffer_element.value[j]));
          }
        }
      }
      return Status::OK();
    }

    Status RestoreInternal(IteratorContext* ctx,
                           IteratorStateReader* reader) override {
      mutex_lock parent_l(parent_mu_);
      mutex_lock l(mu_);
      buffer_.clear();
      TF_RETURN_IF_ERROR(RestoreInput(ctx, reader, input_impl_));
      size_t buffer_size;
      {
        int64 temp;
        TF_RETURN_IF_ERROR(reader->ReadScalar(full_name("buffer_size"), &temp));
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
            TF_RETURN_IF_ERROR(reader->ReadScalar(
                full_name(strings::StrCat("buffer[", i, "].size")), &temp));
            value_size = static_cast<size_t>(temp);
          }
          buffer_element.value.reserve(value_size);
          for (size_t j = 0; j < value_size; j++) {
            buffer_element.value.emplace_back();
            TF_RETURN_IF_ERROR(reader->ReadTensor(
                full_name(strings::StrCat("buffer[", i, "][", j, "]")),
                &buffer_element.value.back()));
          }
        }
      }
      return Status::OK();
    }

   private:
    // A buffer element comprises a status and (if that status is
    // OK) a vector of tensors, representing an element of the input dataset.
    struct BufferElement {
      // The producer sets `status` if getting the input element fails.
      Status status;
      // The buffered data element.
      std::vector<Tensor> value;
      int64 created_us;
    };

    Status Consume(IteratorContext* ctx, std::vector<Tensor>* out_tensors,
                   bool* end_of_sequence) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      //VLOG(0) << "prefetch_dataset_op::Consume";
      const auto& stats_aggregator = ctx->stats_aggregator();
      if (stats_aggregator) {
        stats_aggregator->AddToHistogram(
            stats_utils::BufferUtilizationHistogramName(dataset()->node_name()),
            {static_cast<float>(buffer_.size()) /
             static_cast<float>(auto_tuner_.buffer_limit())},
            num_elements());
        stats_aggregator->AddScalar(
            stats_utils::BufferSizeScalarName(dataset()->node_name()),
            static_cast<float>(buffer_.size()), num_elements());
        stats_aggregator->AddScalar(
            stats_utils::BufferCapacityScalarName(dataset()->node_name()),
            static_cast<float>(auto_tuner_.buffer_limit()), num_elements());
      }
      // A new element is available. Forward the status from computing it, and
      // (if we successfully got an element) the output values.
      Status s = buffer_.front().status;
      if (s.ok()) {
        if (dataset()->slack_period_ > 0 &&
            (num_elements() + 1) % dataset()->slack_period_ == 0) {
          // TODO(rachelim): Consider doing something more sophisticated
          // to decide how long to sleep for; e.g. using a kalman filter.
          int64 slack_us =
              Env::Default()->NowMicros() - buffer_.front().created_us;
          // Every slack_period_-th element, update the most recent slack time,
          // measured by the duration between when the element is prefetched
          // and when it is consumed. We add kSleepFactor * slack_us_ to the
          // measurement because we slept for that duration before prefetching
          // the element.
          slack_us_ = kSleepFactor * slack_us_ + slack_us;
          //VLOG(0) << "Setting slack_us_: " << slack_us_;
          VLOG(2) << "Setting slack_us_: " << slack_us_;
        }
        *out_tensors = std::move(buffer_.front().value);
        RecordBufferDequeue(ctx, *out_tensors);
      }
      // autotune in `Consume` to determine the buffer size.
      auto_tuner_.RecordConsumption(buffer_.size());
      buffer_.pop_front();
      
      //debug// VLOG(0) << "buffer size after Consume: " << buffer_.size();
      // count the step we extract the data from the prefetch stage.
      num_steps_ += 1; // num_steps = generate + consume.

      //VLOG(0) << "-- Consume by " << "Thread " << std::this_thread::get_id() << ", size: " << buffer_.size();

      *end_of_sequence = false;

      // Wake the prefetch thread, in case it has been waiting for space
      // in the buffer. Also wake up threads from other calls to GetNext.
      //
      // TODO(mrry): Consider using different condition variables for
      // GetNext and Prefetch.
      cond_var_.notify_all();
      return s;
    }

    Status EnsurePrefetchThreadStarted(IteratorContext* ctx)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      if (!prefetch_thread_) {
        //VLOG(0) << "EnsurePrefetchThreadStarted;";
        std::shared_ptr<IteratorContext> new_ctx =
            std::make_shared<IteratorContext>(*ctx);
        prefetch_thread_ = ctx->StartThread(
            "tf_data_prefetch", [this, new_ctx]() { PrefetchThread(new_ctx); });
      }
      return Status::OK();
    }

    // Prefetches elements of the input, storing results in an internal
    // buffer.
    //
    // It owns the iterator context passed to it.
    void PrefetchThread(const std::shared_ptr<IteratorContext>& ctx) {
      //VLOG(0) << "PrefetchThread";

      RecordStart(ctx.get());
      auto cleanup = gtl::MakeCleanup([this, ctx] { RecordStop(ctx.get()); });
      // Keep track of where we are in an iteration "burst"
      int num_produced = 0;
      while (true) {
        // 1. Wait for a slot in the buffer.
        {
          mutex_lock l(mu_);
          // auto_tuner_ 是提供一个阈值让 prefetch thread 等待与否.
          // buffer_limit 自然是越大越好. 这样 prefetch thread 不容易停下来.
          while (!cancelled_ && buffer_.size() >= auto_tuner_.buffer_limit()) {
            //VLOG(0) << "PrefetchThread, auto_tuner wait";
            // 几乎就没有怎么进入
            //VLOG(0) << "|| Thread " << std::this_thread::get_id() << ", before, limit: " << auto_tuner_.buffer_limit() << " <= size: " << buffer_.size(); 
            //t_s_limit_ = high_resolution_clock::now();

            RecordStop(ctx.get());
            cond_var_.wait(l);
            RecordStart(ctx.get());
            
            // test timer end. 
            //t_e_limit_ = high_resolution_clock::now();
            //diff_limit_ = duration_cast<milliseconds>(t_e_limit_ - t_s_limit_).count();
            //VLOG(0) << "|| Thread " << std::this_thread::get_id() << " waits milliseconds (reach limit): " << diff_limit_;
            //VLOG(0) << "|| Thread " << std::this_thread::get_id() << ", after, limit: " << auto_tuner_.buffer_limit() << " <= size: " << buffer_.size();             
          }

          if (cancelled_) {
            return;
          }
        }

        // print the buffer size
        //VLOG(0) << "buffer size: " << buffer_.size();

        if (dataset()->slack_period_ > 0 &&
            num_produced % dataset()->slack_period_ == 0) {
          // For the first element in the "burst", sleep for a bit if there is
          // slack.
          VLOG(2) << "Sleeping for: " << slack_us_ * kSleepFactor;
          ctx->env()->SleepForMicroseconds(slack_us_ * kSleepFactor);
        }

        // 2. Read the next element.
        // Acquire the parent lock since we will be reading an element
        // from the input iterator. Note that we do not wish to release
        // this lock till we have added the fetched element to the
        // `buffer_` else there will be local state that may be missed
        // by SaveInternal.
        mutex_lock parent_l(parent_mu_);
        bool end_of_sequence;
        BufferElement buffer_element;

        //VLOG(0) << "++ Thread " << std::this_thread::get_id() << " before get, size: " << buffer_.size(); 
        // timer starts
        //t_s_ = high_resolution_clock::now();
        // Get the buffer element.
        buffer_element.status = input_impl_->GetNext(
            ctx.get(), &buffer_element.value, &end_of_sequence);
        // timer ends
        //t_e_ = high_resolution_clock::now();
        //diff_ = duration_cast<milliseconds>(t_e_ - t_s_).count();
        //VLOG(0) << "++ Thread " << std::this_thread::get_id() << " waits milliseconds to get next data: " << diff_;
        //VLOG(0) << "++ Thread " << std::this_thread::get_id() << " after get, size: " << buffer_.size()+1;

        if (buffer_element.status.ok() 
            && end_of_sequence) {
        //orig// if (buffer_element.status.ok() && end_of_sequence) {
          mutex_lock l(mu_);
          prefetch_thread_finished_ = true;
          cond_var_.notify_all();
          return;
        }

        // 3. Signal that the element has been produced.
        {
          mutex_lock l(mu_);
          RecordBufferEnqueue(ctx.get(), buffer_element.value);
          buffer_element.created_us = ctx->env()->NowMicros();

          // count the training step.
          num_steps_ += 1; // num_steps_ = generate + consume.
          
          // =======================
          // idea: 水塘抽样（Reservoir sampling）
          // =======================
          //VLOG(0) << "- DEBUG 1 -";
          //VLOG(0) << "num_batches_: " << num_batches_;
          if (echoing_buffer_.size() < echo_size_) {
            echoing_buffer_.push_back(buffer_element);
            //VLOG(0) << "push to sampling pool, size = " << echoing_buffer_.size();
          } else {
            // the problem is fresh data can hardly replace new data later.
            //int r = rand() % echo_size_*3; // 1 of 3 chance to replace 

            // each epoch, we should reset the r from 1..40036
            int r = rand() % (num_batches_ % 4000);
            if (r < echo_size_) {
              //VLOG(0) << "echo buffer size = " << echoing_buffer_.size();
              //VLOG(0) << "r = " << r;
              //VLOG(0) << "replace r = " << r;
              echoing_buffer_[r] = buffer_element;
            }
          }

          // num of fresh mini-batches.
          num_batches_ += 1;

          // reset the echoing_buffer_ each epoch, here we hardcore the batch size = 32.
          if (num_batches_ % 4000 == 0) {
            echoing_buffer_.clear();
          }

          //o// // echo 频率
          //o// if (num_batches_ >= echo_size_ && (num_batches_ % echo_freq == 0)) {
          //o//   buffer_.push_back(std::move(buffer_element));
          //o//   for (auto &elem: echoing_buffer_) {
          //o//     buffer_.push_back(elem);
          //o//   }
          //o//   num_produced += (echo_size_ + 1);
          //o// } else {
          //o//   buffer_.push_back(std::move(buffer_element));
          //o//   num_produced += 1;
          //o// }

          // =======================
          // idea: data echoing
          // idea: 能不能一次取 x4.
          // follow up idea: 能不能存一个 buffer 记住之前的某几个, 留着备用?
          //   然后定期更新这些 buffer 内记住的那几个. 为了有混淆效果的 data echoing.
          // push to the buffer_ x 4.
          // =======================

          buffer_.push_back(std::move(buffer_element));

          // cache the previous dataset elements and randomize it.
          // echo size limit is the history to maintain.

          // TODO : design a elegant and flexible way to store history data.
          // history_span for replacement.
          // push_back the current and erase which one?
          // a degree to control the echo_size_ memory(history)
          // 保鲜度指数: new 一点 还是 陈腐一点.
          // --------------------------------------------------------
          // append stale data every "echo_freq of newly generated data steps".          

          //old// if (num_batches % stale_data_sampling_freq == 0) {
          //old//   echoing_buffer_.push_back(buffer_element);
          //old//   //VLOG(0) << "size of echoing_buffer_: " << echoing_buffer_.size();
          //old//   if (echoing_buffer_.size() > echo_size_) {
          //old//     echoing_buffer_.erase(echoing_buffer_.begin());

          //old//     // 我们是否可以 erase 一半呢?
          //old//     //echoing_buffer_.erase(echoing_buffer_.begin() + echoing_buffer_.size() / 2, echoing_buffer_.end());

          //old//     // 我们是否可以 erase 3/4 呢?
          //old//     //echoing_buffer_.erase(echoing_buffer_.begin() + echoing_buffer_.size()*(1/2), echoing_buffer_.end());

          //old//     // 我们是否可以 erase 老一半呢?
          //old//     //echoing_buffer_.erase(echoing_buffer_.begin(), echoing_buffer_.end() - echoing_buffer_.size()*(1/2));
          //old//     
          //old//     // 是否可以全部 erase?
          //old//     //echoing_buffer_.clear();
          //old//   }
          //old// }

          //old// // we append stale data into the fresh data every echo_freq (e.g., 8 new mini-batches) as a new buffer.
          //old// if (num_steps % echo_freq == 0) {
          //old//   // shuffle echoing_buffer_ elements
          //old//   auto rng = std::default_random_engine {};

          //old//   // shuffle(order) 替换成
          //old//   // pick one then replacement => permutation
          //old//   std::shuffle(std::begin(echoing_buffer_), std::end(echoing_buffer_), rng);

          //old//   // push fresh data first, then the echoed data to buffer_ next.
          //old//   buffer_.push_back(buffer_element);
          //old//   //debug// VLOG(0) << "buffer size +1 : " << buffer_.size() 
          //old//   //debug//         << "; echoing_buffer_ size: " << echoing_buffer_.size()
          //old//   //debug//         << "; num of batches: " << num_batches;
          //old//   // push the echoed data from history
          //old//   for (auto &elem: echoing_buffer_) {
          //old//     buffer_.push_back(elem);
          //old//   }
          //old// }
          // --------------------------------------------------------

          //buffer_.push_back(std::move(buffer_element));
          //debug// VLOG(0) << "buffer size +1 : " << buffer_.size() 
          //debug//         << "; echoing_buffer_ size: " << echoing_buffer_.size()
          //debug//         << "; num of batches: " << num_batches;

          // print the buffer size
          //debug// VLOG(0) << "buffer size without pushing echoing_buffer_: " << buffer_.size();

          cond_var_.notify_all();
        }
        ++num_produced;

        // push to the buffer_ xN.
        //VLOG(0) << "PrefetchThread::num_produced: " << num_produced;

      }
    }

    Status WriteStatus(IteratorStateWriter* writer, size_t index,
                       const Status& status) EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      TF_RETURN_IF_ERROR(writer->WriteScalar(
          CodeKey(index), static_cast<int64>(status.code())));
      if (!status.ok()) {
        TF_RETURN_IF_ERROR(writer->WriteScalar(ErrorMessageKey(index),
                                               status.error_message()));
      }
      return Status::OK();
    }

    Status ReadStatus(IteratorStateReader* reader, size_t index, Status* status)
        EXCLUSIVE_LOCKS_REQUIRED(mu_) {
      int64 code_int;
      TF_RETURN_IF_ERROR(reader->ReadScalar(CodeKey(index), &code_int));
      error::Code code = static_cast<error::Code>(code_int);

      if (code != error::Code::OK) {
        string error_message;
        TF_RETURN_IF_ERROR(
            reader->ReadScalar(ErrorMessageKey(index), &error_message));
        *status = Status(code, error_message);
      } else {
        *status = Status::OK();
      }
      return Status::OK();
    }

    string CodeKey(size_t index) {
      return full_name(strings::StrCat("status[", index, "].code"));
    }

    string ErrorMessageKey(size_t index) {
      return full_name(strings::StrCat("status[", index, "].error_message"));
    }

    // This mutex is used to ensure exclusivity between multiple threads
    // reading/writing this iterator's local state.
    mutex mu_;
    // This mutex is used to ensure exclusivity between multiple threads
    // accessing the parent iterator. We keep this separate from `mu_` to
    // allow prefetching to run in parallel with GetNext calls.
    mutex parent_mu_ ACQUIRED_BEFORE(mu_);
    std::unique_ptr<IteratorBase> input_impl_ GUARDED_BY(parent_mu_);
    condition_variable cond_var_;
    PrefetchAutotuner auto_tuner_ GUARDED_BY(mu_);
    std::deque<BufferElement> buffer_ GUARDED_BY(mu_);
    
    // sparated cases: num_steps_ should be private! num_steps_ = consume + generate (insert)
    int num_steps_ = 0;
    // newly generated data, fresh data.
    int num_batches_ = 0;
    // It is used to repeat previous some cached dataset element.
    int echo_size_ = 1600;
    int K_ = 3;
    std::deque<BufferElement> echoing_buffer_ GUARDED_BY(mu_);

    std::unique_ptr<Thread> prefetch_thread_ GUARDED_BY(mu_);
    bool cancelled_ GUARDED_BY(mu_) = false;
    bool prefetch_thread_finished_ GUARDED_BY(mu_) = false;

    std::atomic<int64> slack_us_;

    high_resolution_clock::time_point t_s_;
    high_resolution_clock::time_point t_e_;
    double diff_;

    high_resolution_clock::time_point t_s_limit_;
    high_resolution_clock::time_point t_e_limit_;
    double diff_limit_;

    high_resolution_clock::time_point t_s_empty_;
    high_resolution_clock::time_point t_e_empty_;
    double diff_empty_;
  };
  const DatasetBase* const input_;
  const int64 buffer_size_;

  // If non-zero, determines the period between injecting "slack" into the
  // execution.
  const int64 slack_period_;
};

void PrefetchDatasetOp::MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                                    DatasetBase** output) {
  int64 buffer_size = 0;
  OP_REQUIRES_OK(ctx,
                 ParseScalarArgument<int64>(ctx, "buffer_size", &buffer_size));
  OP_REQUIRES(ctx,
              buffer_size >= 0 || buffer_size == PrefetchAutotuner::kAutoTune,
              errors::InvalidArgument("buffer_size must be >= 0 or set "
                                      "buffer_size to be ",
                                      PrefetchAutotuner::kAutoTune,
                                      " for auto-tuning"));

  if (buffer_size == PrefetchAutotuner::kAutoTune) {
    metrics::RecordTFDataAutotune(kDatasetName);
  }

  *output = new Dataset(ctx, input, buffer_size, slack_period_);
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
