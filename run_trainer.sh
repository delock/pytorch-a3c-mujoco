# start trainers on this node
ENV_NAME=$1

WORLD_SIZE=$2

echo Start $WORLD_SIZE training worker
WORLD_SIZE_MINUS_1=`expr $WORLD_SIZE - 1`

for core_id in `seq 0 $WORLD_SIZE_MINUS_1`
do
    numactl --physcpubind=$core_id python main.py --task train --save_freq 1000 --rank $core_id --env-name $ENV_NAME &
done
