#!/bin/bash

# user_dir="/home/opi"
# user="opi"
user_dir="/root"
user="root"

if [ "$1" != "" ]; then
    hostname=$1

    # Check if device is online
    timeout=2
    ping $hostname -c 1 -W $timeout > /dev/null
    if [ $? == 0 ]; then
        rsync -vrt ./src/yolo_test $user@$hostname:$user_dir/ros2_ws/src
        rsync -vrt ./src/blimp_vision_msgs $user@$hostname:$user_dir/ros2_ws/src
    else
        echo "$hostname is offline :("
    fi
else
    echo "Usage: copyCodeToPi.sh [hostname]"
fi
