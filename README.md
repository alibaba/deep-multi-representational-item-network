# Deep multi-Representational Item NetworK for CTR prediction

Note: we use Python 2.7 and Tensorflow 1.4.

## Prepare Data
    ./prepare_data.sh

When you see the files below, you can do the next work.

* cat_voc.pkl
* mid_voc.pkl
* uid_voc.pkl
* local_train_sample_sorted_by_time
* local_test_sample_sorted_by_time
* reviews-info
* item-info

## Train Model

    mkdir dnn_best_model

    CUDA_VISIBLE_DEVICES=0 python ./script/train.py train [model name]

The model blelow had been supported:
* SVDPP
* Wide (Wide&Deep NN)
* DIN
* DIEN (https://github.com/mouna99/dien)
* DMIN (https://github.com/mengxiaozhibo/DMIN)
* DUMN (https://github.com/hzzai/DUMN)
* DRINK

## Acknowledgements
Our code is implemented based on [DIEN](https://github.com/mouna99/dien), DMIN-CIKM20, TIEN-CIKM20, and DUMN-SIGIR21.
