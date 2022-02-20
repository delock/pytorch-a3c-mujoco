#ENV_NAME=InvertedPendulum-v2
ENV_NAME=$1
NUM_CPUS=`cat /proc/cpuinfo|grep core|grep id|wc -l`

WORLD_SIZE=`expr $NUM_CPUS - 1`
WORLD_SIZE_MINUS_1=`expr $WORLD_SIZE - 1`

echo Start 1 optimizer worker
mkdir -p sync
rm -f sync/*.pt
numactl --physcpubind=`expr $NUM_CPUS - 1` python main.py --task optimize --save_freq 1000 --world_size $WORLD_SIZE --env-name $ENV_NAME &
numactl --physcpubind=`expr $NUM_CPUS - 2` python main.py --task eval --env-name $ENV_NAME --display True &

./run_trainer.sh $ENV_NAME $WORLD_SIZE

read -n1 ans
pkill python
