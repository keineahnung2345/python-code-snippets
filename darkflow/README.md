# darkflow
[thtrieu/darkflow](https://github.com/thtrieu/darkflow)

It converts `.cfg` and `.weights` from darknet into `.meta` and `.pb` of tensorflow.

## Correction
According to [the discussion](https://github.com/thtrieu/darkflow/issues/802#issuecomment-441265886), first open `darkflow/utils/loader.py` and then revise `self.offset = 16` to `self.offset = 20`.

## Installation

```sh
python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Cython
python3 setup.py build_ext --inplace
```

## Convertion
Revise `labels.txt` to the class names your model try to predict.

```sh
python3 ./flow --model <yolov1_or_v2>.cfg --load <yolov1_or_v2>.weights --savepb
```

Then one can find `.meta` and `.pb` files in the direcotry `built_graph`.
