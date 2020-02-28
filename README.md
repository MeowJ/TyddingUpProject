# TyddingUpProject

1. Code for EVA arm:

Enter Automata folder, the scripts and their usage are:

	back_to_initial.py: let EVA move to initial position

	move_arm_ros.py: enter a location and let EVA move to the entered location

	eva_move_camera.py: get the infomation from the camera and let EVA move to the place of the object

	tf_classification_ros.py: subscribe the object location and object label from the object detector, place the object to the container based on object label.


2. Code for computer vision:

	Simuation for colour detection:
	launch colour_detection.launch in panda_simulation package, run start_pose.py and vision.py in the script folder to check the colour detection result.

	Simulation for object classification:
	launch classification_detection.launch in panda_simulation package, run banana_start.py in the script folder and vision_tf_for_simulation.py in Object_detection_trained folder to check the colour detection result.


3. Code for Omnibot simulation:

	Enter ws_moveit, in panda_simulation package, launch omnibot.launch to launch Omnibot into the simulation world 

	run add_collision.py to add a collition in MoveIt and test motion planning 

	change the property of the simulated object: enter box_description/urdf folder and change the property of the red box and banana, play with the values like mass and firction to check different resuls.

	more models of different ojects can be found in models_spawn_library_pkg and spawn_robot_tools_pkg


4. To train the SSD network

	prepare dataset and put them into object-detection_for_training/images (the example dataset is in the test folder)

	run xml_to_csv.py and generate_tfrecord.py 
		
		Example usage:

		python3 generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
  		python3 generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record

	configure the network structure in training/ssd_config.config

	install Tensorflow API

	put the folder under the folder tensorflow/models/tree/master/research/object_detection

	run train.py 
					Example usage:
				    ./train \
				        --logtostderr \
				        --train_dir=path/to/train_dir \
				        --pipeline_config_path=pipeline_config.pbtxt

