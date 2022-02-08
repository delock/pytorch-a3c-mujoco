ENV_NAME=InvertedPendulum-v2
WORLD_SIZE=2

mkdir sync
python main.py --task train    --save_freq 1000 --rank 0 --env-name $ENV_NAME &
python main.py --task train    --save_freq 1000 --rank 1 --env-name $ENV_NAME &
python main.py --task optimize --save_freq 1000 --world_size $WORLD_SIZE --env-name $ENV_NAME &
wait
