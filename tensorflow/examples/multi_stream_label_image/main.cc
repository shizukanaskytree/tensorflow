/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include <fstream>
#include <utility>
#include <vector>
#include <thread>
#include <pthread.h>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

// These are all common classes it's handy to reference with no namespace.
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;
using tensorflow::RunOptions;
using tensorflow::RunMetadata; 

class Inference {

public:
  Inference(
    string image, 
    string graph, 
    int32 input_width, 
    int32 input_height,    
    float input_mean, 
    float input_std, 
    string input_layer,
    string output_layer, 
    bool self_test,
    string root_dir):
  image_(image), 
  graph_(graph), 
  input_width_(input_width), 
  input_height_(input_height),  
  input_mean_(input_mean),
  input_std_(input_std),
  input_layer_(input_layer),
  output_layer_(output_layer),
  self_test_(self_test),
  root_dir_(root_dir) {}


  Status ReadEntireFile(tensorflow::Env* env, const string& filename,
                           Tensor* output);

  Status ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors, int gpriority);

  Status LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session);

  int RunInference(int argc, char* argv[], int gpriority);

private:
  string image_;
  string graph_;
  int32 input_width_;
  int32 input_height_;  
  float input_mean_;
  float input_std_;
  string input_layer_;
  string output_layer_;
  bool self_test_;
  string root_dir_;
};

// Reads a model graph definition from disk, and creates a session object you
// can use to run it.
Status Inference::LoadGraph(const string& graph_file_name,
                 std::unique_ptr<tensorflow::Session>* session) {
  tensorflow::GraphDef graph_def;
  Status load_graph_status =
      ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
  if (!load_graph_status.ok()) {
    return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                        graph_file_name, "'");
  }
  tensorflow::SessionOptions sess_options = tensorflow::SessionOptions();
  sess_options.config.set_use_per_session_threads(false);
  session->reset(tensorflow::NewSession(sess_options));
  Status session_create_status = (*session)->Create(graph_def);
  if (!session_create_status.ok()) {
    return session_create_status;
  }
  return Status::OK();
}


// Given an image file name, read in the data, try to decode it as an image,
// resize it to the requested size, and then scale the values as desired.
Status Inference::ReadTensorFromImageFile(const string& file_name, const int input_height,
                               const int input_width, const float input_mean,
                               const float input_std,
                               std::vector<Tensor>* out_tensors, int gpriority) {
  auto root = tensorflow::Scope::NewRootScope();
  using namespace ::tensorflow::ops;  // NOLINT(build/namespaces)

  string input_name = "file_reader";
  string output_name = "normalized";

  // read file_name into a tensor named input
  Tensor input(tensorflow::DT_STRING, tensorflow::TensorShape());
  TF_RETURN_IF_ERROR(
      ReadEntireFile(tensorflow::Env::Default(), file_name, &input));

  // use a placeholder to read input data
  auto file_reader =
      Placeholder(root.WithOpName("input"), tensorflow::DataType::DT_STRING);

  std::vector<std::pair<string, tensorflow::Tensor>> inputs = {
      {"input", input},
  };

  // Now try to figure out what kind of file it is and decode it.
  const int wanted_channels = 1;
  tensorflow::Output image_reader;
  if (tensorflow::str_util::EndsWith(file_name, ".png")) {
    image_reader = DecodePng(root.WithOpName("png_reader"), file_reader,
                             DecodePng::Channels(wanted_channels));
  } else if (tensorflow::str_util::EndsWith(file_name, ".gif")) {
    // gif decoder returns 4-D tensor, remove the first dim
    image_reader =
        Squeeze(root.WithOpName("squeeze_first_dim"),
                DecodeGif(root.WithOpName("gif_reader"), file_reader));
  } else if (tensorflow::str_util::EndsWith(file_name, ".bmp")) {
    image_reader = DecodeBmp(root.WithOpName("bmp_reader"), file_reader);
  } else {
    // Assume if it's neither a PNG nor a GIF then it must be a JPEG.
    image_reader = DecodeJpeg(root.WithOpName("jpeg_reader"), file_reader,
                              DecodeJpeg::Channels(wanted_channels));
  }
  // Now cast the image data to float so we can do normal math on it.
  auto float_caster =
      Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
  // The convention for image ops in TensorFlow is that all images are expected
  // to be in batches, so that they're four-dimensional arrays with indices of
  // [batch, height, width, channel]. Because we only have a single image, we
  // have to add a batch dimension of 1 to the start with ExpandDims().
  auto dims_expander = ExpandDims(root, float_caster, 0);
  // Bilinearly resize the image to fit the required dimensions.
  auto resized = ResizeBilinear(
      root, dims_expander,
      Const(root.WithOpName("size"), {input_height, input_width}));
  // Subtract the mean and divide by the scale.
  Div(root.WithOpName(output_name), Sub(root, resized, {input_mean}),
      {input_std});

  // This runs the GraphDef network definition that we've just constructed, and
  // returns the results in the output tensor.
  tensorflow::GraphDef graph;
  TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

  tensorflow::SessionOptions sess_options = tensorflow::SessionOptions();
  sess_options.config.set_use_per_session_threads(false);
  std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(sess_options));

  RunOptions options;
  RunMetadata run_metadata;
  TF_RETURN_IF_ERROR(session->Create(graph));
  
  if (gpriority == 1) {
    options.set_gpriority(1);  
    TF_RETURN_IF_ERROR(session->Run(options, {inputs}, {output_name}, {}, out_tensors, &run_metadata));
  } else {
    options.set_gpriority(0);  
    TF_RETURN_IF_ERROR(session->Run(options, {inputs}, {output_name}, {}, out_tensors, &run_metadata));
  }
  return Status::OK();
}

Status Inference::ReadEntireFile(tensorflow::Env* env, const string& filename,
                             Tensor* output) {
  tensorflow::uint64 file_size = 0;
  TF_RETURN_IF_ERROR(env->GetFileSize(filename, &file_size));

  string contents;
  contents.resize(file_size);

  std::unique_ptr<tensorflow::RandomAccessFile> file;
  TF_RETURN_IF_ERROR(env->NewRandomAccessFile(filename, &file));

  tensorflow::StringPiece data;
  TF_RETURN_IF_ERROR(file->Read(0, file_size, &data, &(contents)[0]));
  if (data.size() != file_size) {
    return tensorflow::errors::DataLoss("Truncated read of '", filename,
                                        "' expected ", file_size, " got ",
                                        data.size());
  }
  output->scalar<string>()() = string(data);
  return Status::OK();
}


int Inference::RunInference(int argc, char* argv[], int gpriority){

  // We need to call this to set up global state for TensorFlow.
  tensorflow::port::InitMain(argv[0], &argc, &argv);
  if (argc > 1) {
    LOG(ERROR) << "Unknown argument " << argv[1] << "\n";
    return -1;
  }

  // First we load and initialize the model.
  std::unique_ptr<tensorflow::Session> session;
  string graph_path = tensorflow::io::JoinPath(root_dir_, graph_);
  Status load_graph_status = LoadGraph(graph_path, &session);
  if (!load_graph_status.ok()) {
    LOG(ERROR) << load_graph_status;
    return -1;
  }

  // Get the image from disk as a float array of numbers, resized and normalized
  // to the specifications the main graph expects.
  std::vector<Tensor> resized_tensors;
  string image_path = tensorflow::io::JoinPath(root_dir_, image_);
  Status read_tensor_status =
      ReadTensorFromImageFile(image_path, input_height_, 
                              input_width_, input_mean_,
                              input_std_, &resized_tensors, gpriority);
  if (!read_tensor_status.ok()) {
    LOG(ERROR) << read_tensor_status;
    return -1;
  }
  const Tensor& resized_tensor = resized_tensors[0];

  // Actually run the image through the model.
  std::vector<Tensor> outputs;
  
  RunOptions options;
  RunMetadata run_metadata;

  Status run_status;
  if (gpriority == 1) {
    options.set_gpriority(1);

    for (int i = 0; i < 4000; i++) {
      run_status = session->Run(options, {{input_layer_, resized_tensor}},
                                     {output_layer_}, {}, &outputs, &run_metadata);
    }

  } else {
    options.set_gpriority(0);

    for (int i = 0; i < 4000; i++) {
      run_status = session->Run(options, {{input_layer_, resized_tensor}},
                                     {output_layer_}, {}, &outputs, &run_metadata);
    }
  }

  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
    return -1;
  }
  return 0;
}

void t1 (int argc, char* argv[]) {
  string image_high = "tensorflow/examples/multi_stream_label_image/data/119_h.png";
  string graph_high = "tensorflow/examples/multi_stream_label_image/data/frozen_graph_h.pb";
  int32 input_width = 28;
  int32 input_height = 28;
  float input_mean = 128;
  float input_std = 128;
  string input_layer = "input_node";
  string output_layer = "output_node";
  bool self_test = false;
  string root_dir = "";
  Inference inf_high = Inference(
    image_high, graph_high, input_width, 
    input_height, input_mean, 
    input_std, input_layer, output_layer, 
    self_test, root_dir);

  inf_high.RunInference(argc, argv, 1);
}


void t2 (int argc, char* argv[]) {
  string image_low = "tensorflow/examples/multi_stream_label_image/data/119.png";
  string graph_low = "tensorflow/examples/multi_stream_label_image/data/frozen_graph.pb";
  int32 input_width = 28;
  int32 input_height = 28;
  float input_mean = 128;
  float input_std = 128;
  string input_layer = "input_node";
  string output_layer = "output_node";
  bool self_test = false;
  string root_dir = "";

  Inference inf_low = Inference(
    image_low, graph_low, input_width, 
    input_height, input_mean, 
    input_std, input_layer, output_layer, 
    self_test, root_dir);

  inf_low.RunInference(argc, argv, 0);

  Inference inf_low_1 = Inference(
    image_low, graph_low, input_width, 
    input_height, input_mean, 
    input_std, input_layer, output_layer, 
    self_test, root_dir);

  inf_low_1.RunInference(argc, argv, 0);
}

void t3 (int argc, char* argv[]) {
  string image_low = "tensorflow/examples/multi_stream_label_image/data/119.png";
  string graph_low = "tensorflow/examples/multi_stream_label_image/data/frozen_graph.pb";
  int32 input_width = 28;
  int32 input_height = 28;
  float input_mean = 128;
  float input_std = 128;
  string input_layer = "input_node";
  string output_layer = "output_node";
  bool self_test = false;
  string root_dir = "";

  Inference inf_low = Inference(
    image_low, graph_low, input_width, 
    input_height, input_mean, 
    input_std, input_layer, output_layer, 
    self_test, root_dir);

  inf_low.RunInference(argc, argv, 0);
}

void t4 (int argc, char* argv[]) {
  string image_low = "tensorflow/examples/multi_stream_label_image/data/119.png";
  string graph_low = "tensorflow/examples/multi_stream_label_image/data/frozen_graph.pb";
  int32 input_width = 28;
  int32 input_height = 28;
  float input_mean = 128;
  float input_std = 128;
  string input_layer = "input_node";
  string output_layer = "output_node";
  bool self_test = false;
  string root_dir = "";

  Inference inf_low = Inference(
    image_low, graph_low, input_width, 
    input_height, input_mean, 
    input_std, input_layer, output_layer, 
    self_test, root_dir);

  inf_low.RunInference(argc, argv, 0);
}


int main(int argc, char* argv[]) {
  // These are the command-line flags the program can understand.
  // They define where the graph and input data is located, and what kind of
  // input the model expects. If you train your own model, or use something
  // other than inception_v3, then you'll need to update these.
/// for mnist   
//  string image_low = "tensorflow/examples/multi_stream_label_image/data/119.png";
//  string image_high = "tensorflow/examples/multi_stream_label_image/data/119_h.png";
//  string graph_low = "tensorflow/examples/multi_stream_label_image/data/frozen_graph.pb";
//  string graph_high = "tensorflow/examples/multi_stream_label_image/data/frozen_graph_h.pb";
//  int32 input_width = 28;
//  int32 input_height = 28;
//  float input_mean = 128;
//  float input_std = 128;
//  string input_layer = "input_node";
//  string output_layer = "output_node";
//  bool self_test = false;
//  string root_dir = "";
 
/// for inception v3
//  string image_low = "tensorflow/examples/multi_stream_label_image/data/grace_hopper_low.jpg";
//  string graph_low = "tensorflow/examples/multi_stream_label_image/data/inception_v3_2016_08_28_frozen_low.pb";
//  int32 input_width = 299;
//  int32 input_height = 299;
//  float input_mean = 0;
//  float input_std = 255;
//  string input_layer = "Placeholder";
//  string output_layer = "InceptionV3/Predictions/Reshape_1";
//  string output_layer = "output";

  std::thread t_1 (t1, argc, argv);
  std::thread t_2 (t2, argc, argv);
  std::thread t_3 (t3, argc, argv);
  std::thread t_4 (t4, argc, argv);

  sched_param sch;
  int policy;
  sch.sched_priority = 90;
  if (pthread_setschedparam(t_1.native_handle(), SCHED_FIFO, &sch)) {
    std::cout << "Failed to setschedparam: " << std::strerror(errno) << '\n';
  }

  t_1.join();
  t_2.join();
  t_3.join();
  t_4.join();

  return 0;
}

