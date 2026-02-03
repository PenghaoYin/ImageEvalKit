set -x
MODEL_NAME=$1
EVAL_ENV=$2

RESULT_DIR="$PWD/output/${MODEL_NAME}"

# export LD_LIBRARY_PATH="$CONDA_BASE/envs/${EVAL_ENV}/lib:${LD_LIBRARY_PATH}"
gedit_dir="benchmarks/gedit/"
echo "gedit directory: $gedit_dir"
cd $gedit_dir || exit

GEDIT_ASSET=./gedit_asset

$CONDA_BASE/envs/${EVAL_ENV}/bin/python -u step2_gedit_bench.py \
    --model_name ${MODEL_NAME} \
    --save_path ${RESULT_DIR}/gedit/images \
    --backbone gpt4o \
    --source_path ${GEDIT_ASSET}
$CONDA_BASE/envs/${EVAL_ENV}/bin/python -u step3_calculate_statistics.py \
    --model_name ${MODEL_NAME} \
    --save_path ${RESULT_DIR}/gedit/images \
    --backbone gpt4o \
    --language en > ${RESULT_DIR}/gedit/result.txt
cat ${RESULT_DIR}/gedit/result.txt
