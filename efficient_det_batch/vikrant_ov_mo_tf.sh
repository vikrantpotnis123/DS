set -x

if [ $# -ne 1 ]; then
    echo "$0 <d0/d1/d2/d3/d4/d5/d6/d7>"
    exit
fi

MO_ROOT=../openvino/model-optimizer
# https://github.com/google/automl/blob/96e1fee/efficientdet/hparams_config.py#L304
if [ "$1" = "d0" ]; then
    INPUT_SIZE=512
elif [ "$1" = "d1" ]; then
    INPUT_SIZE=640
elif [ "$1" = "d2" ]; then
    INPUT_SIZE=768
elif [ "$1" = "d3" ]; then
    INPUT_SIZE=896
elif [ "$1" = "d4" ]; then
    INPUT_SIZE=1024
elif [ "$1" = "d5" ]; then
    INPUT_SIZE=1280
elif [ "$1" = "d6" ];then
    INPUT_SIZE=1280
elif [ "$1" = "d7" ];then
    INPUT_SIZE=1536
fi
BATCH_SIZE=4
NUM_CHANNELS=3
INPUT_SHAPE="[1,$INPUT_SIZE,$INPUT_SIZE,$NUM_CHANNELS]" # d5, d6

echo "INPUT_SHAPE = ${INPUT_SHAPE}"

# LOG_LEVEL=DEBUG
LOG_LEVEL=ERROR
TRANS_CONFIG=$MO_ROOT/extensions/front/tf/automl_efficientdet.json
INPUT_MODEL=../savedmodel_${1}/efficientdet-${1}_frozen.pb
python3 $MO_ROOT/mo_tf.py \
--input_model $INPUT_MODEL \
--transformations_config $TRANS_CONFIG \
--input_shape $INPUT_SHAPE
