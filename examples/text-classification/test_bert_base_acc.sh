export TASK_NAME=sst2

python run_glue_private.py \
  --model_name_or_path andeskyl/bert-base-cased-$TASK_NAME \
  --task_name $TASK_NAME \
  --max_length 128 \
  --acc \
  --per_device_eval_batch_size 1 \
  --output_dir eval_private/$TASK_NAME/
