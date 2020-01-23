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

#include <string>

#include "tensorflow/core/common_runtime/session_factory.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/public/session.h"

namespace tensorflow {

Session::Session() {}

Session::~Session() {}

Status Session::Run(const RunOptions& run_options,
                    const std::vector<std::pair<string, Tensor> >& inputs,
                    const std::vector<string>& output_tensor_names,
                    const std::vector<string>& target_node_names,
                    std::vector<Tensor>* outputs, RunMetadata* run_metadata) {
  return errors::Unimplemented(
      "Run with options is not supported for this session.");
}

Status Session::PRunSetup(const std::vector<string>& input_names,
                          const std::vector<string>& output_names,
                          const std::vector<string>& target_nodes,
                          string* handle) {
  return errors::Unimplemented(
      "Partial run is not supported for this session.");
}

Status Session::PRun(const string& handle,
                     const std::vector<std::pair<string, Tensor> >& inputs,
                     const std::vector<string>& output_names,
                     std::vector<Tensor>* outputs) {
  return errors::Unimplemented(
      "Partial run is not supported for this session.");
}

Session* NewSession(const SessionOptions& options) {

  SessionFactory* factory;
  Status s = SessionFactory::GetFactory(options, &factory);

  if (!s.ok()) {
    LOG(ERROR) << s;
    return nullptr;
  }
  
  Session* out_session;
  s = NewSession(options, &out_session);

  if (!s.ok()) {
    LOG(ERROR) << "Failed to create session: " << s;
    return nullptr;
  }
  return out_session;
}

/** \brief Create a new Session subject to Session options.
 *
 *  \param[in] options: SessionOptions& ;
 *         SessionOptions is used to provide 1. environment used; 2. target
 *         used to perform all computations according to host:port. 3. A bunch
 *         of Session configuration parameters.
 *
 *  \param[out] out_session: Session* ;
 *         A Session instance lets a caller drive a TensorFlow graph computation
 *
 *  \return Status
 *
 *  \details
 *          When a Session is created with a given target, a new Session object
 *          is bound to the universe of resources specified by that target.
 *          Those resources are available to this session to perform
 *          computation described in the GraphDef.  After extending the session
 *          with a graph, the caller uses the Run() API to perform the
 *          computation and potentially fetch outputs as Tensors.
 *
 *  \note The more higher the API is, the more easy and clear to use it.
 *        In this case, input the `options`, the output is a session.
 *        Whoever calls it, who will get the desired output result.
 *        And I don't need to care about the implementation when I only review
 *        it. I only need to know when and who calls it, what I will get.
 */
Status NewSession(const SessionOptions& options, Session** out_session) {
  SessionFactory* factory;
  Status s = SessionFactory::GetFactory(options, &factory);
  if (!s.ok()) {
    *out_session = nullptr;
    LOG(ERROR) << s;
    return s;
  }
  s = factory->NewSession(options, out_session);
  if (!s.ok()) {
    *out_session = nullptr;
  }
  return s;
}

Status Reset(const SessionOptions& options,
             const std::vector<string>& containers) {
  SessionFactory* factory;
  TF_RETURN_IF_ERROR(SessionFactory::GetFactory(options, &factory));
  return factory->Reset(options, containers);
}

}  // namespace tensorflow
