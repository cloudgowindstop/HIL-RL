export PYTHONPATH="/media/data_convertor/lerobot:$PYTHONPATH"
export PYTHONPATH="/media/data_convertor/lerobot/lerobot:$PYTHONPATH"
unset http_proxy https_proxy




# python convert_to_cosmos.py \
#   --input_dir /media/hermine_dataset/franka1/franka1_h5data_convert/insert_usb \
#   --output_dir /media/hermine_dataset/franka1/franka1_h5data_convert_cosmos \
#   --demo_ratio 0.5 \
#   --limit 60



  python convert_to_cosmos.py \
  --input_dir /media/hermine_dataset/franka1/franka1_h5data_convert/insert_usb \
  --output_dir /media/hermine_dataset/franka1/franka1_h5data_convert_cosmos \
  --demo_ratio 1 \
  --limit 60