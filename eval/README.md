# 评测脚本编写规范

评测脚本由三部分组成：
1. 读取参数，包括待评测模型的名称和评测时使用的环境
2. 设置环境变量，例如设置图片存放路径、用于评测的模型权重存储的路径，以及切换conda等
3. 切换工作目录，将目录切换到评测仓库的根目录
4. 执行评测命令

例如下面脚本（genai.sh）：
```bash
# 读取参数
MODEL_NAME=$1
EVAL_ENV=$2

# 设置环境变量
RESULT_DIR="$PWD/output/${MODEL_NAME}"
VISION_TOWER="$PWD/eval_models/clip-vit-large-patch14-336"
T5_PATH="$PWD/eval_models/clip-flant5-xxl"
source utils/use_cuda.sh 11.8

# 切换工作目录
genai_dir="$PWD/benchmarks/genai"
echo "genai 目录: $genai_dir"
cd $genai_dir || exit

# 执行评测命令
META_DIR="eval_prompts/genai1600"
IMAGE_DIR="${RESULT_DIR}/genai/images"
VISION_TOWER=${VISION_TOWER} $CONDA_BASE/envs/yph-genai/bin/python -m step2_run_model \
    --model_path ${T5_PATH} \
    --image_dir ${IMAGE_DIR} \
    --meta_dir ${META_DIR} > ${RESULT_DIR}/genai/results.txt
cat ${RESULT_DIR}/genai/results.txt
```