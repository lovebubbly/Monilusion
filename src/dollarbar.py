import pandas as pd
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import os
import multiprocessing as mp # multiprocessing 추가
import numba # numba 추가

# --------------------------------------------------------------------------
# Helper 함수들 (기존과 동일)
# --------------------------------------------------------------------------

def detect_column_mappings(df: pd.DataFrame) -> Dict[str, str]:
    """
    🔍 데이터프레임의 컬럼명을 자동 감지하고 매핑합니다.
    """
    column_mapping = {
        'timestamp': None,
        'price': None,
        'volume': None
    }
    columns = df.columns.tolist()

    timestamp_candidates = ['timestamp', 'time', 'datetime', 'date', 'ts', 'trade_time']
    for col in columns:
        if any(candidate in col.lower() for candidate in timestamp_candidates):
            column_mapping['timestamp'] = col
            break
    
    price_candidates = ['price', 'close', 'last_price', 'px']
    for col in columns:
        if any(candidate in col.lower() for candidate in price_candidates):
            column_mapping['price'] = col
            break
    
    volume_candidates = ['volume', 'vol', 'qty', 'quantity', 'size', 'amount', 'base_volume']
    for col in columns:
        if any(candidate in col.lower() for candidate in volume_candidates):
            column_mapping['volume'] = col
            break
    
    return column_mapping

# --------------------------------------------------------------------------
# ✨ 최적화된 핵심 함수들 ✨
# --------------------------------------------------------------------------

def load_and_prepare_chunk_for_worker(chunk_files: List[str], column_mapping: Dict[str, str]) -> Optional[pd.DataFrame]:
    """
    🔥 [Worker Process용] ThreadPoolExecutor를 사용한 안전한 파일 로딩 및 기본 전처리
    - 이 함수는 각 병렬 프로세스 내에서 실행됩니다.
    """
    dfs = []
    # 각 워커 내에서는 I/O 작업에 효율적인 스레드 풀 사용
    with ThreadPoolExecutor(max_workers=4) as executor:
        required_cols = [col for col in column_mapping.values() if col is not None]
        futures = {executor.submit(pd.read_parquet, f, columns=required_cols): f for f in chunk_files}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None and not result.empty:
                    dfs.append(result)
            except Exception:
                pass # 에러는 메인 프로세스에서 감지
    
    if not dfs:
        return None

    combined_df = pd.concat(dfs, ignore_index=True)
    
    ts_col, price_col, vol_col = column_mapping['timestamp'], column_mapping['price'], column_mapping['volume']

    if not all(col in combined_df.columns for col in [ts_col, price_col, vol_col]):
        return None
        
    combined_df[ts_col] = pd.to_datetime(combined_df[ts_col], unit='ms')
    combined_df[price_col] = pd.to_numeric(combined_df[price_col], errors='coerce')
    combined_df[vol_col] = pd.to_numeric(combined_df[vol_col], errors='coerce')
    combined_df.dropna(subset=[price_col, vol_col], inplace=True)
    
    combined_df['dollar_volume'] = combined_df[price_col] * combined_df[vol_col]
    
    return combined_df

def process_chunk_for_volume_analysis(args: Tuple[List[str], Dict[str, str]]) -> Optional[pd.Series]:
    """
    🔥 [Worker Process용] 파일 청크를 받아 일별 거래대금으로 리샘플링하는 함수
    """
    chunk_files, column_mapping = args
    prepared_df = load_and_prepare_chunk_for_worker(chunk_files, column_mapping)
    if prepared_df is not None and not prepared_df.empty:
        ts_col = column_mapping['timestamp']
        return prepared_df.set_index(ts_col)['dollar_volume'].resample('D').sum()
    return None

def analyze_daily_dollar_volume_parallel(all_files: List[str]) -> Optional[pd.Series]:
    """
    🚀 [메모리 최적화] Multiprocessing을 사용하여 전체 파일의 일별 총 거래대금을 병렬로 분석합니다.
    """
    tqdm.write("Step 1: 일별 거래대금 병렬 분석을 시작합니다...")
    
    try:
        sample_df = pd.read_parquet(all_files[0])
        column_mapping = detect_column_mappings(sample_df)
        if not all(col for col in column_mapping.values()):
            raise ValueError("필수 컬럼(timestamp, price, volume)을 자동 감지하지 못했습니다.")
        tqdm.write(f"🎯 컬럼 매핑 성공: {column_mapping}")
    except Exception as e:
        tqdm.write(f"❌ 초기 컬럼 매핑 실패: {e}")
        return None

    num_cores = mp.cpu_count()
    tqdm.write(f"✅ 사용 가능 CPU 코어: {num_cores}개. 병렬 처리를 시작합니다.")

    num_chunks = min(len(all_files), num_cores * 4) 
    chunk_size = (len(all_files) + num_chunks - 1) // num_chunks
    file_chunks = [all_files[i:i + chunk_size] for i in range(0, len(all_files), chunk_size)]
    
    # [메모리 최적화] 결과를 리스트에 모두 쌓지 않고, 점진적으로 합산합니다.
    total_daily_volume = pd.Series(dtype=np.float64)
    
    with mp.Pool(processes=num_cores) as pool:
        tasks = [(chunk, column_mapping) for chunk in file_chunks]
        
        with tqdm(total=len(tasks), desc="📊 병렬로 거래대금 계산 중") as pbar:
            for result in pool.imap_unordered(process_chunk_for_volume_analysis, tasks):
                if result is not None and not result.empty:
                    # [메모리 최적화] 스트리밍 방식으로 결과 합산 (메모리 사용량 대폭 감소)
                    total_daily_volume = total_daily_volume.add(result, fill_value=0)
                pbar.update(1)

    if total_daily_volume.empty:
        tqdm.write("⚠️ 분석할 데이터가 없습니다.")
        return None
        
    tqdm.write("🔄 병렬 처리 결과 합산 완료!")
    return total_daily_volume.sort_index()

# --------------------------------------------------------------------------
# 추천 및 시각화 함수 (기존과 동일)
# --------------------------------------------------------------------------

def get_threshold_recommendations(daily_dollar_volume: pd.Series, target_bars_per_day: int) -> Tuple[float, float, float]:
    """
    🎯 분석된 일별 거래대금과 목표 바 개수를 기반으로 최적 임계값을 추천합니다.
    """
    tqdm.write("\nStep 2 & 3: 최적 임계값 계산을 시작합니다...")
    
    full_period_avg = daily_dollar_volume.mean()
    bullish_market_avg = daily_dollar_volume.quantile(0.75)
    bearish_market_avg = daily_dollar_volume.quantile(0.25)
    
    recommended_threshold = full_period_avg / target_bars_per_day if target_bars_per_day > 0 else 0
    bullish_threshold = bullish_market_avg / target_bars_per_day if target_bars_per_day > 0 else 0
    bearish_threshold = bearish_market_avg / target_bars_per_day if target_bars_per_day > 0 else 0
    
    return recommended_threshold, bullish_threshold, bearish_threshold

def plot_volume_analysis(daily_dollar_volume: pd.Series, recommended_threshold: float, target_bars_per_day: int):
    """
    🎨 분석 결과를 시각화하여 보여줍니다.
    """
    sns.set_theme(style="whitegrid", palette="viridis")
    fig, ax1 = plt.subplots(figsize=(16, 8))
    
    ax1.set_title(f'일별 거래대금 분석 및 추천 임계값 (하루 목표 바: {target_bars_per_day}개)', fontsize=18, pad=20, weight='bold')
    ax1.plot(daily_dollar_volume.index, daily_dollar_volume, label='일별 거래대금', color='deepskyblue', alpha=0.7, zorder=2)
    ax1.set_xlabel('날짜', fontsize=12)
    ax1.set_ylabel('거래대금 (USD)', fontsize=12, color='deepskyblue')
    ax1.tick_params(axis='y', labelcolor='deepskyblue')
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax1.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e9:.1f}B'))

    rolling_avg = daily_dollar_volume.rolling(window=30).mean()
    ax1.plot(rolling_avg.index, rolling_avg, color='mediumblue', linestyle='--', label='30일 이동평균', zorder=3)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    expected_bars = daily_dollar_volume / recommended_threshold if recommended_threshold > 0 else pd.Series(0, index=daily_dollar_volume.index)
    ax2.plot(expected_bars.index, expected_bars, label='예상 일일 바 개수', color='darkorange', alpha=0.6, linestyle=':', zorder=1)
    ax2.set_ylabel('예상 바 개수 (개)', fontsize=12, color='darkorange')
    ax2.tick_params(axis='y', labelcolor='darkorange')
    ax2.axhline(y=target_bars_per_day, color='red', linestyle='--', label=f'목표 바 개수 ({target_bars_per_day}개)', zorder=4)
    ax2.legend(loc='upper right')

    fig.tight_layout(pad=1.5)
    plt.show()

# --------------------------------------------------------------------------
# 🚀 메인 실행부
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Windows에서 multiprocessing 사용 시 필수
    if os.name == 'nt':
        mp.freeze_support()
    
    # --- 1. 설정 ---
    DATA_FOLDER = Path("C:/Monilusion/daily_cache")
    TARGET_BARS_PER_DAY = 100 
    
    # --- 2. 파일 목록 가져오기 ---
    all_files = sorted([str(f) for f in DATA_FOLDER.glob("*.parquet")])
    if not all_files:
        print(f"❌ '{DATA_FOLDER}' 경로에 parquet 파일이 없습니다. 경로를 확인해주세요.")
    else:
        print(f"✅ 총 {len(all_files)}개의 데이터 파일을 발견했습니다.")
        
        # --- 3. 최적화된 분석 실행 ---
        daily_volume_df = analyze_daily_dollar_volume_parallel(all_files)
        
        if daily_volume_df is not None and not daily_volume_df.empty:
            
            # --- 4. 임계값 추천 ---
            rec_thresh, bull_thresh, bear_thresh = get_threshold_recommendations(daily_volume_df, TARGET_BARS_PER_DAY)
            
            # --- 5. 결과 출력 및 시각화 ---
            print("\n" + "="*50)
            print("� 달러 바 임계값 병렬 분석 결과 (메모리 최적화) 🚀")
            print("="*50)
            print(f"🎯 하루 평균 목표 바 개수: {TARGET_BARS_PER_DAY} 개")
            print(f"💰 전체 기간 일평균 거래대금: ${daily_volume_df.mean():,.0f}")
            print("\n--- 추천 임계값 ---")
            print(f"⭐ 균형 임계값 (전체 기간 평균 기반): ${rec_thresh:,.0f}")
            print(f"🔥 활황장 임계값 (상위 25% 기간 기반): ${bull_thresh:,.0f}")
            print(f"❄️ 침체장 임계값 (하위 25% 기간 기반): ${bear_thresh:,.0f}")
            print("="*50)
            print("\n💡 제안:")
            print(f"1. 시작점: 우선 '균형 임계값'인 ${rec_thresh:,.0f}을(를) 너의 달러 바 생성 스크립트에 적용해봐.")
            print("2. 전략 세분화: 만약 시장 국면(활황/침체)을 미리 판단할 수 있다면, 해당 국면에 맞는 임계값을 사용하는 동적 전략도 고려해볼 수 있어.")
            print("3. 시각화 확인: 아래 차트를 보고 추천 임계값 적용 시 일별 바 개수가 어떻게 변동하는지 확인해봐. (목표선에 근접하는지)")
            
            plot_volume_analysis(daily_volume_df, rec_thresh, TARGET_BARS_PER_DAY)