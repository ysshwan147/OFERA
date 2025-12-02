@echo off
set MODEL_NAME=FateAvatar
set CONFIG_PATH=.\config\fateavatar.yaml
set DATASET_PATH=.\data\insta\ysh
set EXP_DIR=.\workspace\insta\ysh
set EXP_NAME=ysh

set CKPT_PATH=.\workspace\insta\ysh\baking\edit\checkpoints_edited\mount3.pth

REM python train_mono_avatar.py --model_name %MODEL_NAME% --config %CONFIG_PATH% --root_path %DATASET_PATH% ^
REM     --workspace %EXP_DIR% --name %EXP_NAME%

REM python train_generate_pseudo.py --model_name %MODEL_NAME% --config %CONFIG_PATH% --workspace %EXP_DIR% ^
REM     --name %EXP_NAME%

REM python train_full_avatar.py --model_name %MODEL_NAME% --config %CONFIG_PATH% --root_path %DATASET_PATH% ^
REM     --workspace %EXP_DIR% --name %EXP_NAME%

REM python train_neural_baking.py --config %CONFIG_PATH% --root_path %DATASET_PATH% ^
REM     --workspace %EXP_DIR% --name %EXP_NAME%

REM python avatar_edit_baked.py ^
REM     --config %CONFIG_PATH% ^
REM     --root_path %DATASET_PATH% ^
REM     --workspace %EXP_DIR% ^
REM     --name %EXP_NAME%
    REM --use_full_head_resume

python avatar_gui.py ^
    --model_name %MODEL_NAME% ^
    --config %CONFIG_PATH% ^
    --workspace %EXP_DIR% ^
    --name %EXP_NAME% ^
    --use_baked_resume ^
    --ckpt_path %CKPT_PATH%
    REM --use_full_head_resume
