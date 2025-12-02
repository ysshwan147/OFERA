SUBJECT="empty"

if [ -n "$1" ]; then SUBJECT=${1}; fi

MODEL_NAME="FateAvatar"
CONFIG_PATH="./config/fateavatar.yaml"
DATASET_PATH="./data/insta/$SUBJECT"
EXP_DIR="./workspace/insta/$SUBJECT"
EXP_NAME="fateavatar_insta_$SUBJECT"

python train_mono_avatar.py --model_name $MODEL_NAME --config $CONFIG_PATH --root_path $DATASET_PATH \
    --workspace $EXP_DIR --name $EXP_NAME

python train_generate_pseudo.py --model_name $MODEL_NAME --config $CONFIG_PATH --workspace $EXP_DIR \
    --name $EXP_NAME

python train_full_avatar.py --model_name $MODEL_NAME --config $CONFIG_PATH --root_path $DATASET_PATH \
    --workspace $EXP_DIR --name $EXP_NAME

python train_neural_baking.py --config $CONFIG_PATH --root_path $DATASET_PATH \
    --workspace $EXP_DIR --name $EXP_NAME

python train_neural_baking.py --config $CONFIG_PATH --root_path $DATASET_PATH \
    --workspace $EXP_DIR --name $EXP_NAME --use_full_head_resume


python avatar_edit_baked.py \
    --config $CONFIG_PATH \
    --root_path $DATASET_PATH \
    --workspace $EXP_DIR \
    --name $EXP_NAME
    # --use_full_head_resume

python avatar_gui.py \
	--model_name $MODEL_NAME \
	--config $CONFIG_PATH \
	--workspace $EXP_DIR \
	--name $EXP_NAME \
    --use_baked_resume
	# --ckpt_path $CKPT_PATH \
	# --use_full_head_resume \
	
# python avatar_gui.py --model_name "FateAvatar" --config "./config/fateavatar.yaml" --workspace "./workspace/insta/ysh_mlp" --name "ysh_mlp" --use_baked_resume --use_full_head_resume