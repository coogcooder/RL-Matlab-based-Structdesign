obs_num1 = 7 
obs_num2 = 9
acts_num1=1
acts_num2 = 6

HIDDEN_SIZE1 = 100
HIDDEN_SIZE2 = 100


EPOCH_NUM = 10000                              # エポック数
MEMORY_SIZE = 2000                            # メモリサイズいくつで学習を開始するか
BATCH_SIZE = 200                              # バッチサイズ
EPSILON1 = 0.6                               
EPSILON2 = 0.6 
EPSILON_DECREASE = 0.0003                     # εの減少値
EPSILON_MIN = 0.2                              # εの下限
TRAIN_FREQ = 1                                 # 学習間隔
UPDATE_TARGET_Q_FREQ = 5                       # QNet更新間隔
GAMMA = 0.9                                   # 割引率
LOG_FREQ = 5               # ログ出力の間隔
LOG =10