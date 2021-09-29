output_dir=models/activitynet
config_file=$output_dir/config.yml

python test_net.py --config-file $config_file OUTPUT_DIR $output_dir 