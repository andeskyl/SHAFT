export TASK_NAME=qnli

python run_glue_private.py \
  --model_name_or_path andeskyl/bert-large-cased-$TASK_NAME \
  --task_name $TASK_NAME \
  --len_data 128 \
  --max_length 128 \
  --comp \
  --per_device_eval_batch_size 1 \
  --output_dir eval_private/$TASK_NAME/
