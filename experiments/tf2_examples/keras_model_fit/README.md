Post:
https://www.tensorflow.org/guide/keras/train_and_evaluate

How to run?

```
python main.py
```


```
(hm) wxf@protago-hp01-3090:~/tf2_prj/example_code/keras_model_fit$ python main.py 
2021-11-18 22:25:26.569322: W tensorflow/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2021-11-18 22:25:26.569339: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
2021-11-18 22:25:27.584309: E tensorflow/stream_executor/cuda/cuda_driver.cc:271] failed call to cuInit: CUDA_ERROR_NO_DEVICE: no CUDA-capable device is detected
2021-11-18 22:25:27.584347: I tensorflow/stream_executor/cuda/cuda_diagnostics.cc:156] kernel driver does not appear to be running on this host (protago-hp01-3090): /proc/driver/nvidia/version does not exist
2021-11-18 22:25:27.584676: I tensorflow/core/platform/cpu_feature_guard.cc:151] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX2 FMA
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Fit model on training data
Epoch 1/2
782/782 [==============================] - 1s 796us/step - loss: 0.3431 - sparse_categorical_accuracy: 0.9031 - val_loss: 0.1850 - val_sparse_categorical_accuracy: 0.9456
Epoch 2/2
782/782 [==============================] - 1s 693us/step - loss: 0.1608 - sparse_categorical_accuracy: 0.9510 - val_loss: 0.1514 - val_sparse_categorical_accuracy: 0.9546
history.history: {'loss': [0.34313446283340454, 0.16083578765392303], 'sparse_categorical_accuracy': [0.9030600190162659, 0.9510400295257568], 'val_loss': [0.18495462834835052, 0.15143175423145294], 'val_sparse_categorical_accuracy': [0.9455999732017517, 0.9545999765396118]}
Evaluate on test data
79/79 [==============================] - 0s 473us/step - loss: 0.1472 - sparse_categorical_accuracy: 0.9582
test loss, test acc: [0.14722983539104462, 0.9581999778747559]
Generate predictions for 3 samples
predictions shape: (3, 10)
```