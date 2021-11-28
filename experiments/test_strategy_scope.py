import tensorflow as tf
from transformers import (
    TFTrainingArguments,
    HfArgumentParser,
    TFAutoModelForMaskedLM,
    AutoConfig,
)

import debugpy

debugpy.listen(5678)
debugpy.wait_for_client()
debugpy.breakpoint()

from scope_tf import _setup_strategy

parser = HfArgumentParser((TFTrainingArguments,))
(training_args,) = parser.parse_args_into_dataclasses()

# my_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
# my_variable = tf.Variable(my_tensor)


# https://www.tensorflow.org/guide/variable
# scope_var = training_args.strategy.scope() # 下面用自己写的 strategy 取代了 transformer 里面的.
scope_var = _setup_strategy()


checkpoint = None
model_name_or_path = "distilbert-base-cased"
config = AutoConfig.from_pretrained(model_name_or_path)
with scope_var:
    model = TFAutoModelForMaskedLM.from_pretrained(model_name_or_path, config=config)
    model.compile()
