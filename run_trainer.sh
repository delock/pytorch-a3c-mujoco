# start trainers on this node
ENV_NAME=$1
NUM_CPUS=`cat /proc/cpuinfo|grep core|grep id|sort -u|wc -l`

WORLD_SIZE=$NUM_CPUS
echo Start $WORLD_SIZE training worker
WORLD_SIZE_MINUS_1=`expr $WORLD_SIZE - 1`

for core_id in `seq 0 $WORLD_SIZE_MINUS_1`
do
    numactl --physcpubind=$core_id python main.py --task train --save_freq 1000 --rank $core_id --display True --env-name $ENV_NAME &
done
