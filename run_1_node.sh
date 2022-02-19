#ENV_NAME=InvertedPendulum-v2
ENV_NAME=$1
NUM_CPUS=`cat /proc/cpuinfo|grep core|grep id|sort -u|wc -l`

WORLD_SIZE=$NUM_CPUS
echo Start 1 optimizer worker and $WORLD_SIZE training worker
WORLD_SIZE_MINUS_1=`expr $WORLD_SIZE - 1`

mkdir -p sync
rm -f sync/*.pt
numactl --physcpubind=0 python main.py --task optimize --save_freq 1000 --world_size $WORLD_SIZE --env-name $ENV_NAME &

for core_id in `seq 0 $WORLD_SIZE_MINUS_1`
do
    numactl --physcpubind=$core_id python main.py --task train --save_freq 1000 --rank $core_id --display True --env-name $ENV_NAME &
done
wait
