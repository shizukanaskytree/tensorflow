/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/scanner.h"

namespace tensorflow {
namespace strings {

void Scanner::ScanUntilImpl(char end_ch, bool escaped) {
  for (;;) {
    if (cur_.empty()) {
      Error();
      return;
    }
    const char ch = cur_[0];
    if (ch == end_ch) {
      return;
    }

    cur_.remove_prefix(1);
    if (escaped && ch == '\\') {
      // Escape character, skip next character.
      if (cur_.empty()) {
        Error();
        return;
      }
      cur_.remove_prefix(1);
    }
  }
}

bool Scanner::GetResult(StringPiece* remaining, StringPiece* capture) {

  // 1.
  // 输入输出
  // remaining: input
  // capture: output

  if (error_) {
    return false;
  }
  if (remaining != nullptr) {
    *remaining = cur_;
    // 1.
    // p *remaining
    // "y: resource"

    // 2.
    // p cur_
    // "resource"

  }
  if (capture != nullptr) {
    const char* end = capture_end_ == nullptr ? cur_.data() : capture_end_;
    // 1.
    // p capture_end_
    // ": resource"

    // 2.
    // p *end
    // ':'

    *capture = StringPiece(capture_start_, end - capture_start_);
    // 1.
    // p *capture_start_
    // 'y'

    // 1.1
    // p capture_start_
    // "y: resource"

    // 2.
    // p *end
    // ':'

    // 3.
    // p end-capture_start_
    // 1

    // 4.
    // p *capture
    // "y: resource", length_ = 1
  }
  return true;
}

// 1.
// where am I ?
//
// bool ConsumeInOutName(StringPiece* sp, StringPiece* out) {
//   return Scanner(*sp)
//       .One(Scanner::LOWERLETTER)
//       .Any(Scanner::LOWERLETTER_DIGIT_UNDERSCORE)
//       .StopCapture()
//       .AnySpace()
//       .OneLiteral(":")
//       .AnySpace()
//       .GetResult(sp, out);
// }
//
// Thread #1 [xla_kernel_crea] 31989 [core: 15] (Suspended : Step)
// 	tensorflow::(anonymous namespace)::ConsumeInOutName at op_def_builder.cc:285 0x7ffff628495d
// 	tensorflow::(anonymous namespace)::FinalizeInputOrOutput at op_def_builder.cc:347 0x7ffff6284cc7
// 	tensorflow::OpDefBuilder::Finalize() at op_def_builder.cc:645 0x7ffff6286ed9
// 	tensorflow::FunctionDefHelper::Define() at function.cc:1,830 0x7ffff621f986
// 	tensorflow::XTimesY() at xla_kernel_creator_test.cc:54 0x555555f13c80
// 	tensorflow::XlaKernelCreatorTest_OneFloatOneResourceArgument_Test::TestBody() at xla_kernel_creator_test.cc:97 0x555555f1461f
// 	testing::internal::HandleSehExceptionsInMethodIfSupported<testing::Test, void>() at gtest.cc:2,424 0x7fffad262c76
// 	testing::internal::HandleExceptionsInMethodIfSupported<testing::Test, void>() at gtest.cc:2,460 0x7fffad25e2ed
// 	testing::Test::Run() at gtest.cc:2,499 0x7fffad24c64e
// 	testing::TestInfo::Run() at gtest.cc:2,675 0x7fffad24cfe3
// 	<...more frames...>
//

}  // namespace strings
}  // namespace tensorflow
