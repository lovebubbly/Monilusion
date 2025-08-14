import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from scipy.stats import skew

# Advanced feature libraries
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

# Scikit-learn & XGBoost
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

# Pandas TA for base features
import pandas_ta as ta

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


class FinancialDataPipeline:
    """
    v10: 최종 진화 - 딥러닝(Transformer) 모델을 위한 데이터 준비 및 시퀀싱 파이프라인.
    XGBoost 훈련 모듈을 제거하고, 딥러닝 모델 학습에 최적화된 시퀀스 데이터를 생성하는 역할에 집중.

    - Purpose: To prepare and sequence data for advanced deep learning models.
    - Goal-Oriented Labeling: 미래 수익률과 변동성을 기준으로 국면을 명확하게 정의.
    - State-of-the-Art Features: 다중 시간대, 웨이블릿, 변동성 스퀴즈 등 모든 피처를 생성.
    - Output: .npy files (X_train, y_train, X_val, y_val) ready for PyTorch/TensorFlow.
    """
    def __init__(self, config):
        self.config = config
        self.feature_scaler = StandardScaler()
        self.regime_map = None

        if not os.path.exists(self.config['output_dir']):
            os.makedirs(self.config['output_dir'])
            print(f"✅ Created output directory: {self.config['output_dir']}")

    def _load_and_prep_data(self, df_path, resample_freq):
        print(f"Step 1: 💾 '{df_path}'에서 데이터 로딩 및 전처리...")
        df = pd.read_parquet(df_path)
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.set_index('open_time').sort_index()
        
        if resample_freq:
            print(f"     - 🔬 데이터를 '{resample_freq}' 주기로 리샘플링...")
            agg_rules = {'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum', 'buy_value':'sum', 'sell_value':'sum'}
            df = df.resample(resample_freq).agg(agg_rules).dropna()
        return df

    def _create_state_of_the_art_features(self, df, mtf_config):
        print("Step 2: 🧠 최첨단 피처 엔지니어링...")
        features_df = df.copy()
        
        features_df.ta.ema(length=10, append=True)
        features_df.ta.ema(length=30, append=True)
        features_df.ta.squeeze(lazy=False, detailed=False, append=True)
        features_df['chop'] = features_df.ta.chop()
        features_df['log_return'] = np.log(features_df['close'] / features_df['close'].shift(1))
        features_df['price_volatility_20'] = features_df['log_return'].rolling(20).std()
        features_df['relative_range'] = (features_df['high'] - features_df['low']) / features_df['close']
        features_df['volume_surge'] = features_df['volume'] / features_df['volume'].rolling(50).mean()
        
        if mtf_config:
            for freq in mtf_config:
                df_htf = df.resample(freq).agg({'close':'last', 'volume':'sum'}).dropna()
                df_htf[f'ema_50_{freq}'] = df_htf.ta.ema(length=50)
                df_htf[f'rsi_14_{freq}'] = df_htf.ta.rsi()
                features_df = pd.merge_asof(left=features_df, right=df_htf[[f'ema_50_{freq}', f'rsi_14_{freq}']], left_index=True, right_index=True, direction='backward')

        if PYWT_AVAILABLE and 'log_return' in features_df:
            try:
                coeffs = pywt.wavedec(features_df['log_return'].dropna(), 'db4', level=3)
                coeffs[1:] = [pywt.threshold(c, value=0.1, mode='soft') for c in coeffs[1:]]
                denoised_signal = pywt.waverec(coeffs, 'db4')
                features_df['log_return_denoised'] = pd.Series(denoised_signal[:len(features_df['log_return'].dropna())], index=features_df['log_return'].dropna().index)
            except Exception as e:
                print(f"       - 웨이블릿 변환 오류: {e}")

        return features_df.dropna()

    def _define_regimes_by_future_outcome(self, df, horizon, n_bins):
        print("Step 3: 🎯 목표 기반 국면 '정의'...")
        df_labeled = df.copy()
        
        future_returns = df_labeled['close'].pct_change(horizon).shift(-horizon)
        future_volatility = df_labeled['log_return'].rolling(horizon).std().shift(-horizon)
        
        df_labeled['return_q'] = pd.qcut(future_returns, n_bins, labels=False, duplicates='drop')
        df_labeled['vol_q'] = pd.qcut(future_volatility, n_bins, labels=False, duplicates='drop')
        df_labeled.dropna(subset=['return_q', 'vol_q'], inplace=True)
        
        regime_conditions = [
            (df_labeled['return_q'] == n_bins - 1) & (df_labeled['vol_q'] >= n_bins // 2),
            (df_labeled['return_q'] == n_bins - 1) & (df_labeled['vol_q'] < n_bins // 2),
            (df_labeled['return_q'] == 0) & (df_labeled['vol_q'] >= n_bins // 2),
            (df_labeled['return_q'] == 0) & (df_labeled['vol_q'] < n_bins // 2),
            (df_labeled['vol_q'] == n_bins - 1) & (df_labeled['return_q'] > 0) & (df_labeled['return_q'] < n_bins - 1),
        ]
        regime_labels = [0, 1, 2, 3, 4]
        self.regime_map = {0:"🚀 변동성 상승", 1:"📈 안정적 상승", 2:"🚨 변동성 하락", 3:"📉 안정적 하락", 4:"횡보"}
        df_labeled['regime'] = np.select(regime_conditions, regime_labels, default=4).astype(int)
        
        print("     - 국면 정의 완료:")
        regime_dist = df_labeled['regime'].value_counts(normalize=True).sort_index()
        regime_dist.index = regime_dist.index.map(self.regime_map)
        print(regime_dist)
        return df_labeled

    def _split_and_prepare_sequences(self, df):
        print("Step 4: 🔪 시계열 누수 방지 데이터 분할 및 시퀀스 생성...")
        
        non_feature_cols = ['open', 'high', 'low', 'close', 'volume', 'buy_value', 'sell_value', 'regime', 'return_q', 'vol_q']
        features = [col for col in df.columns if col not in non_feature_cols and not col.startswith('future')]
        
        X = df[features]
        y = df['regime']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.config['random_state'], stratify=y)
        X_train_scaled = pd.DataFrame(self.feature_scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test_scaled = pd.DataFrame(self.feature_scaler.transform(X_test), index=X_test.index, columns=X_test.columns)

        # 훈련/검증 데이터 분리 (X_test를 val로 사용)
        X_val_scaled, y_val = X_test_scaled, y_test
        
        print(f"     - 분할 결과: Train={len(X_train_scaled)}, Validation={len(X_val_scaled)}")

        def create_and_save(scaled_data, labels, name, seq_len):
            if len(scaled_data) < seq_len:
                print(f"     - ⚠️ 경고: '{name}' 데이터셋이 시퀀스 길이({seq_len})보다 작아 생성을 건너뜁니다.")
                return
            
            X_seq, y_seq = [], []
            for i in tqdm(range(len(scaled_data) - seq_len + 1), desc=f"       - {name} 시퀀스 생성 중"):
                X_seq.append(scaled_data.iloc[i:i + seq_len].values)
                y_seq.append(labels.iloc[i + seq_len - 1])
            
            X_seq, y_seq = np.array(X_seq), np.array(y_seq)
            
            np.save(os.path.join(self.config['output_dir'], f'X_{name}.npy'), X_seq)
            np.save(os.path.join(self.config['output_dir'], f'y_{name}.npy'), y_seq)
            print(f"     - {name} Set 저장 완료: X={X_seq.shape}, y={y_seq.shape}")

        seq_len = self.config['sequence_length']
        create_and_save(X_train_scaled, y_train, 'train', seq_len)
        create_and_save(X_val_scaled, y_val, 'val', seq_len)
        
    def _save_artifacts(self):
        print("Step 5: 💾 스케일러 등 최종 산출물 저장...")
        
        scaler_path = os.path.join(self.config['output_dir'], 'feature_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.feature_scaler, f)
        print(f"     - 스케일러 저장 완료: {scaler_path}")
        
    def run(self):
        print("🚀 딥러닝용 데이터 준비 파이프라인 가동 (v10)!")
        print("="*60)
        
        base_df = self._load_and_prep_data(self.config['input_path'], self.config['resample_freq'])
        features_df = self._create_state_of_the_art_features(base_df, self.config.get('mtf_freqs'))
        labeled_df = self._define_regimes_by_future_outcome(features_df, self.config['prediction_horizon'], self.config['n_quantile_bins'])
        self._split_and_prepare_sequences(labeled_df)
        self._save_artifacts()
        
        print("\n🎉 모든 파이프라인 성공적으로 완료! 🎉")
        print(f"결과물은 '{self.config['output_dir']}' 디렉토리에 저장되었습니다.")


if __name__ == '__main__':
    CONFIG = {
        'input_path': 'data/dollar_bars_BTCUSDT_2021-2025.parquet',
        'output_dir': 'data/processed_for_transformer_v10',
        'resample_freq': '4H', 
        'mtf_freqs': ['1D', '3D'],
        'prediction_horizon': 10,
        'n_quantile_bins': 5,
        'sequence_length': 64, # Transformer 모델에 입력될 시퀀스 길이
        'random_state': 42,
    }

    if not os.path.exists(CONFIG['input_path']):
        print(f"Warning: Input file not found. Creating dummy data...")
        os.makedirs('data', exist_ok=True)
        dates = pd.to_datetime(pd.date_range('2021-01-01', '2025-03-31', freq='h'))
        price = 20000 + np.random.randn(len(dates)).cumsum() * 100
        volume = np.random.randint(100, 1000, size=len(dates))
        dummy_df = pd.DataFrame({'open_time': dates, 'close': price, 'volume': volume, 'buy_value': (volume * price / 2) * (1 + np.random.rand(len(dates)) * 0.1), 'sell_value': (volume * price / 2) * (1 - np.random.rand(len(dates)) * 0.1), 'open': price, 'high': price, 'low': price})
        dummy_df.to_parquet(CONFIG['input_path'])
        print("Dummy data created.")

    pipeline = FinancialDataPipeline(config=CONFIG)
    pipeline.run()