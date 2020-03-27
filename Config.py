OPTIMIRZER_ = ["opt_SGD", "opt_Momentum", "opt_RMSprop", "opt_Adam"]

FLAG_RESTORE_MODEL = True
PHASE = 'Train'  # Predict
MODEL_NAME = r'Models/aligner_epoch_175.pt'
MODEL_SAVE_PATH = r'Models'
RESULTS_LOG_PATH = r'Results'
IMG_TO_PREDICT = r'G:\Coding\FaceKeypointsDetection\Test\4.png'

DEVICE = 'gpu'  # gpu
EPOCH = 300
BATCH_TRAIN = 20
BATCH_VAL = 70
OPTIMIRZER = OPTIMIRZER_[3]
LEARNING_RATE = 0.0005
LOG_INTERVAL = 20
SAVE_MODEL_INTERVAL = 5
NET_IMG_SIZE = (112, 112)

