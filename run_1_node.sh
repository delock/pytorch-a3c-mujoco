#ENV_NAME=InvertedPendulum-v2
ENV_NAME=$1
NUM_CPUS=`cat /proc/cpuinfo|grep core|grep id|sort -u|wc -l`

WORLD_SIZE=$NUM_CPUS
WORLD_SIZE_MINUS_1=`expr $WORLD_SIZE - 1`

echo Start 1 optimizer worker
mkdir -p sync
rm -f sync/*.pt
numactl --physcpubind=0 python main.py --task optimize --save_freq 1000 --world_size $WORLD_SIZE --env-name $ENV_NAME &

./run_trainer.sh $ENV_NAME

read -n1 ans
pkill python
