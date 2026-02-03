MODEL_NAME=$1
EVAL_ENV=$2

RESULT_DIR="$PWD/output/${MODEL_NAME}"

t2icompbench_dir="benchmarks/T2I-CompBench"
echo "t2icompbench directory: $t2icompbench_dir"
cd $t2icompbench_dir || exit

bash eval_all.sh $RESULT_DIR/t2icompbench $EVAL_ENV