CKPT_PATH="/home/wz3008/steve/logs/2025-10-07T04:16:59.250397_4096_movie"

python eval_fgari_video.py \
    --data_path "/home/wz3008/dataset/movi-e-eval/*" \
    --trained_model_paths \
    ${CKPT_PATH}/best_model.pt \
    --use_dvae \
    --vocab_size 4096 \
    --batch_size 64

