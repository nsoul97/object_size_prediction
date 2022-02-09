
WINDOW = 10                                 # the number of frames of the window

START_STD = 2.0                             # the wrist y-standard deviation value that is used to identify the start of a grasping movement
STOP_STD = 1.5                              # the wrist y-standard deviation value that is used to identify the end of a grasping movement

PROB_THRESHOLD = 0.6                        # the minimum probability value of non-noisy OpenPose keypoint detections
DIST_THRESHOLD = 10                         # the maximum x-axis and y-axis distance of non-noisy OpenPose keypoint detections from their nearest neighbouring frame.

MOV_COMPLETION = [20, 40, 60, 80, 100]      # the movement completion percentages of the grasping movements for which a model is evaluated


FS_NAMES = {0: ['TI-Ap'],                   # The features that correspond to each feature set id
            1: ['TI-Ap', 'TM-Ap', 'IM-Ap'],
            2: ['TI-Ap', 'TM-Ap', 'IM-Ap', 'wrist x', 'wrist y'],
            3: ['TI-Ap', 'TM-Ap', 'IM-Ap', 'wrist x-std_dev', 'wrist y-std_dev'],
            4: ['TI-Ap', 'TM-Ap', 'IM-Ap', 'wrist xy-std_dist'],
            5: ['TI-Ap', 'TM-Ap', 'IM-Ap', 'wrist x-speed', 'wrist y-speed'],
            6: ['TI-Ap', 'TM-Ap', 'IM-Ap', 'wrist xy-speed'],
            7: ['TI-Ap', 'TM-Ap', 'IM-Ap', 'wrist x', 'wrist y', 'wrist x-std_dev', 'wrist y-std_dev'],
            8: ['TI-Ap', 'TM-Ap', 'IM-Ap', 'wrist x', 'wrist y', 'wrist x-speed', 'wrist y-speed']}