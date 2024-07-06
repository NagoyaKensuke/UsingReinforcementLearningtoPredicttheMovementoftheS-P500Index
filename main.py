import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import gym
from gym import spaces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# FRED APIキーの設定
os.environ['FRED_API_KEY'] = 'seacret_key'

# 環境変数からFRED APIキーを取得
fred_api_key = os.getenv('FRED_API_KEY')

# FREDクライアントの初期化
fred = Fred(api_key=fred_api_key)

# マクロ経済データの取得関数
def get_macro_data():
    gdp = fred.get_series('GDP')
    cpi = fred.get_series('CPIAUCSL')
    unemployment_rate = fred.get_series('UNRATE')
    return pd.concat([gdp, cpi, unemployment_rate], axis=1, keys=['GDP', 'CPI', 'Unemployment Rate'])

# 株価データの取得関数
def get_stock_data():
    sp500 = fred.get_series('SP500')
    nasdaq = fred.get_series('NASDAQCOM')
    return pd.concat([sp500, nasdaq], axis=1, keys=['S&P500', 'Nasdaq'])

# データの取得と前処理
macro_data = get_macro_data()
stock_data = get_stock_data()
data = pd.concat([macro_data, stock_data], axis=1).dropna()

# S&P500の実際の価格を保存
actual_sp500 = data['S&P500'].copy()

# データの前処理
data = data.pct_change().dropna()
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# 強化学習環境の定義
class StockPredictionEnv(gym.Env):
    def __init__(self, data):
        super(StockPredictionEnv, self).__init__()
        self.data = data
        self.current_step = 0

        # アクション空間を連続値に変更
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # 観測空間の定義
        obs_shape = (data.shape[1],)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self.data.iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.current_step = 0

        # 報酬を予測精度に基づいて設計
        actual_return = self.data['S&P500'].iloc[self.current_step]
        reward = -np.abs(action[0] - actual_return)  # 予測と実際の値の差の絶対値を最小化

        done = self.current_step == len(self.data) - 1
        return self.data.iloc[self.current_step].values.astype(np.float32), reward, done, {}

# データの分割（訓練データとテストデータ）
train_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)

# 環境の作成
env = DummyVecEnv([lambda: StockPredictionEnv(train_data)])
env = VecNormalize(env, norm_obs=True, norm_reward=True)

# 強化学習モデル（PPO）の訓練
model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=50000)

# テストデータでの予測
obs = env.reset()
predictions = []
for i in range(len(test_data)):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    predictions.append(action[0])

# 予測を実際のS&P500価格にスケーリング
scaler = MinMaxScaler()
actual_sp500_test = actual_sp500.iloc[-len(test_data):]
scaler.fit(actual_sp500_test.values.reshape(-1, 1))
scaled_predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

# 予測結果の可視化
plt.figure(figsize=(16, 8))
plt.plot(test_data.index, actual_sp500_test, label='Actual S&P500', color='blue')
plt.plot(test_data.index, scaled_predictions, label='Predicted S&P500', color='red')
plt.title('S&P500 Price Prediction')
plt.xlabel('Date')
plt.ylabel('S&P500 Price')
plt.legend()
plt.tight_layout()
plt.savefig('sp500_prediction_results.png', dpi=300)
plt.show()

# 評価指標の計算
mse = np.mean((actual_sp500_test.values - scaled_predictions)**2)
mae = np.mean(np.abs(actual_sp500_test.values - scaled_predictions))
print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
