# data_root="/mnt/afs/zhenjianan/data/teleoperation"
# data_root="/mnt/afs/lichanglin/dev/temp/test_data"
data_root="/mnt/afs/public/data/teleop_data_dual"
# data_root="/mnt/afs/public/data/teleop_data_v2"
output_root="/mnt/afs/public/data/lerobot_data"
# output_root="/mnt/project/groceryvla/dataset/lerobot/2_1b"
object_name=$1
echo "$object_name"
python /mnt/afs/lichanglin/dev/vla_network/vla_network/dataset/scripts/convert_tele2.2_data_to_lerobotv2.1_double.py \
	--data_dir "$data_root/$object_name" \
	--object_name=$object_name \
	--output_root=$output_root \
	--cut_data \
    --robot_type 2.2_wb \
    --whole_body \
	--with_bbox \
	--with_goal \
	--incremental \
	# --debug_path /mnt/afs/lichanglin/dev/temp/debug \
	# --incremental \
	