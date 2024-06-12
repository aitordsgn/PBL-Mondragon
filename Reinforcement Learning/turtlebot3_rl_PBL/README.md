Go to mu-gym folder and install mu-gym environments


git clone -b dashing-devel https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git

TODO: Save the updated version of the turtlebot3_simulations and turtlebot in GitHub

# Install turtlebot gym
- Fo to the mu-gym folder and execute
```
pip install -e .

```

# Environment variable

- Export the GAZEBO environement variable (Models and Plugins). The links used in the instruction can not be the same in your computer. You can include them in your .bashrc. Remenber to source once you have changed.

```
TOBE UPDATED export GAZEBO_MODEL_PATH=~/mu-eps/ros_ws/src/M5/m5_franka_gazebo/models:${GAZEBO_MODEL_PATH}
```
Plugins

Compile the plugin

More info in Gazebo Plugins -> https://classic.gazebosim.org/tutorials?tut=plugins_hello_world&cat=write_plugin

```
export GAZEBO_PLUGIN_PATH=/home/user/turtlebot3_ws/src/turtlebot3/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_dqn_world/obstacle_plugin/build/:${GAZEBO_PLUGIN_PATH}
```

- Source
```
alias python=python3
source /opt/ros/dashing/setup.bash
source ~/turtlebot3_ws/install/setup.bash
source ~/venv/bin/activate
```

# Test

- Run your prefered worlds
```

```

- Train
```
python test_PDQN.py
```

# View results in Tensorboard
- Run tensorboard server
```
tensorboard --logdir=tensorboard_session --bind_all
```
- View the results in Firefox or your prefered navigator (the IP is just an example)
```
http://192.168.233.135:6060/
```


