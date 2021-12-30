
```
Traceback (most recent call last):
  File "test_dist_allreduce.py", line 3, in <module>
    test = tf.distribute.CollectiveAllReduceStrategyAtom()
AttributeError: module 'tensorflow._api.v2.distribute' has no attribute 'CollectiveAllReduceStrategyAtom'
```

```
tensorflow/_api/v2/distribute
```


```
grep -wnr "collective_all_reduce_strategy" --exclude="*test*.py" --exclude="*.py" --exclude="*DOC*" --exclude="*.pbtxt"
```




```
tensorflow/BUILD:1274:    output_dir = "_api/v2/",
```

```
gen_api_init_files(
    name = "tf_python_api_gen_v2",
    srcs = [
        "api_template.__init__.py",
        "compat_template.__init__.py",
        "compat_template_v1.__init__.py",
    ],
    api_version = 2,
    compat_api_versions = [
        1,
        2,
    ],
    compat_init_templates = [
        "compat_template_v1.__init__.py",
        "compat_template.__init__.py",
    ],
    output_dir = "_api/v2/",
    output_files = TENSORFLOW_API_INIT_FILES_V2,
    output_package = "tensorflow._api.v2",
    root_file_name = "v2.py",
    root_init_template = "$(location api_template.__init__.py)",
)
```


依赖关系 - all path:


```
(hm) wxf@seir19:~/tf2/tensorflow$
bazel query "allpaths(//tensorflow:tf_python_api_gen_v2, //tensorflow/python/distribute:collective_all_reduce_strategy)"
```

```
(hm) wxf@seir19:~/tf2/tensorflow$ bazel query "allpaths(//tensorflow:tf_python_api_gen_v2, //tensorflow/python/distribute:collective_all_reduce_strategy)"
//tensorflow:tf_python_api_gen_v2
//tensorflow:create_tensorflow.python_api_tf_python_api_gen_v2
//tensorflow/python:modules_with_exports
//tensorflow/python:no_contrib
//tensorflow/python/tpu:tpu_noestimator
//tensorflow/python/tpu:feature_column_v2
//tensorflow/python/tpu:feature_column
//tensorflow/python/distribute:strategy_combinations
//tensorflow/python/distribute:test_util
//tensorflow/python/distribute:distribute
//tensorflow/python/distribute/experimental:experimental
//tensorflow/python/distribute:combinations
//tensorflow/python:keras_lib
//tensorflow/python/keras:keras
//tensorflow/python/keras/wrappers:wrappers
//tensorflow/python/keras/utils:utils
//tensorflow/python/keras/preprocessing:preprocessing
//tensorflow/python/keras/utils:all_utils
//tensorflow/python/keras/utils:multi_gpu_utils
//tensorflow/python/keras/mixed_precision:mixed_precision_experimental
//tensorflow/python/keras/mixed_precision:get_layer_policy
//tensorflow/python/keras/applications:applications
//tensorflow/python/keras:testing_utils
//tensorflow/python/keras:engine
//tensorflow/python/keras:models
//tensorflow/python/feature_column:feature_column_py
//tensorflow/python:rnn
//tensorflow/python:rnn_cell
//tensorflow/python/keras/layers/legacy_rnn:rnn_cell_impl
//tensorflow/python:layers
//tensorflow/python/layers:layers
//tensorflow/python/keras/legacy_tf_layers:pooling
//tensorflow/python/keras/legacy_tf_layers:normalization
//tensorflow/python/keras/legacy_tf_layers:core
//tensorflow/python/keras/legacy_tf_layers:convolutional
//tensorflow/python/keras/layers:layers
//tensorflow/python/keras/layers/preprocessing:preprocessing
//tensorflow/python/keras/layers/preprocessing:text_vectorization
//tensorflow/python/keras/layers/preprocessing:string_lookup
//tensorflow/python/keras/layers/preprocessing:reduction
//tensorflow/python/keras/layers/preprocessing:preprocessing_stage
//tensorflow/python/keras/layers/preprocessing:normalization
//tensorflow/python/keras/layers/preprocessing:integer_lookup
//tensorflow/python/keras/layers/preprocessing:index_lookup
//tensorflow/python/keras/layers/preprocessing:category_encoding
//tensorflow/python/keras/layers/preprocessing:image_preprocessing
//tensorflow/python/keras/layers/preprocessing:hashing
//tensorflow/python/keras/layers/preprocessing:discretization
//tensorflow/python/keras/layers/preprocessing:category_crossing
//tensorflow/python/keras/engine:engine
//tensorflow/python/keras/engine:base_preprocessing_layer
//tensorflow/python/keras/layers/normalization:normalization
//tensorflow/python/keras/layers/normalization:layer_normalization
//tensorflow/python/keras/layers/normalization:batch_normalization_v1
//tensorflow/python/keras/layers/normalization:batch_normalization
//tensorflow/python/keras/layers:wrappers
//tensorflow/python/keras/layers:rnn_cell_wrapper_v2
//tensorflow/python/keras/layers:noise
//tensorflow/python/keras/layers:multi_head_attention
//tensorflow/python/keras/layers:merge
//tensorflow/python/keras/layers:local
//tensorflow/python/keras/layers:kernelized
//tensorflow/python/keras/layers:embeddings
//tensorflow/python/keras/layers:einsum_dense
//tensorflow/python/keras/layers:dense_attention
//tensorflow/python/keras/layers:cudnn_recurrent
//tensorflow/python/keras/layers:recurrent_v2
//tensorflow/python/keras/layers:core
//tensorflow/python/keras/layers:convolutional_recurrent
//tensorflow/python/keras/layers:recurrent
//tensorflow/python/keras/layers:convolutional
//tensorflow/python/keras/layers:pooling
//tensorflow/python/keras/feature_column:feature_column
//tensorflow/python/keras/feature_column:sequence_feature_column
//tensorflow/python/keras/feature_column:dense_features_v2
//tensorflow/python/keras/feature_column:dense_features
//tensorflow/python/keras/feature_column:base_feature_layer
//tensorflow/python/feature_column:feature_column_v2
//tensorflow/python/feature_column:feature_column
//tensorflow/python/keras:metrics
//tensorflow/python/keras:activations
//tensorflow/python/keras/layers:advanced_activations
//tensorflow/python/keras:base_layer
//tensorflow/python:layers_base
//tensorflow/python/layers:layers_base
//tensorflow/python/keras/legacy_tf_layers:layers_base
//tensorflow/python/keras/engine:base_layer
//tensorflow/python/keras/mixed_precision:loss_scale_optimizer
//tensorflow/python/distribute:collective_all_reduce_strategy
Loading: 0 packages loaded
(hm) wxf@seir19:~/tf2/tensorflow$
```

依赖关系 - some path:

```
(hm) wxf@seir19:~/tf2/tensorflow$ bazel query "somepath(//tensorflow:tf_python_api_gen_v2, //tensorflow/python/distribute:collective_all_reduce_strategy)"
//tensorflow:tf_python_api_gen_v2
//tensorflow:create_tensorflow.python_api_tf_python_api_gen_v2
//tensorflow/python:no_contrib
//tensorflow/python/distribute:combinations
//tensorflow/python/distribute:collective_all_reduce_strategy
Loading: 0 packages loaded
(hm) wxf@seir19:~/tf2/tensorflow$
```

