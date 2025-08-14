import pandas as pd
import numpy as np
import numba
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from pathlib import Path
from typing import List, Tuple, Optional

def detect_column_mappings(df: pd.DataFrame) -> dict:
    """
    🔍 데이터프레임의 컬럼명을 자동 감지하고 매핑
    """
    column_mapping = {
        'timestamp': None,
        'price': None,
        'volume': None
    }
    
    columns = df.columns.tolist()
    tqdm.write(f"📋 발견된 컬럼들: {columns}")
    
    # timestamp 컬럼 찾기
    timestamp_candidates = ['timestamp', 'time', 'datetime', 'date', 'ts']
    for col in columns:
        if any(candidate in col.lower() for candidate in timestamp_candidates):
            column_mapping['timestamp'] = col
            break
    
    # price 컬럼 찾기
    price_candidates = ['price', 'close', 'last_price', 'px']
    for col in columns:
        if any(candidate in col.lower() for candidate in price_candidates):
            column_mapping['price'] = col
            break
    
    # volume 컬럼 찾기
    volume_candidates = ['volume', 'vol', 'qty', 'quantity', 'size', 'amount', 'base_volume']
    for col in columns:
        if any(candidate in col.lower() for candidate in volume_candidates):
            column_mapping['volume'] = col
            break
    
    # 결과 출력
    tqdm.write(f"🎯 컬럼 매핑 결과:")
    for key, value in column_mapping.items():
        tqdm.write(f"   {key}: {value}")
    
    return column_mapping

@numba.njit
def create_dollar_bars_numba_safe(
    timestamps: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    dollar_threshold: float,
    start_accumulated: float = 0.0,
    start_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """
    🔥 Numba 최적화된 달러 바 생성 함수 (안전성 강화)
    """
    n = len(timestamps)
    
    # 결과 저장용 리스트
    bar_timestamps = []
    bar_opens = []
    bar_highs = []
    bar_lows = []
    bar_closes = []
    
    # 현재 바 상태
    current_dollar_volume = start_accumulated
    current_open = 0.0
    current_high = 0.0
    current_low = 0.0
    current_close = 0.0
    current_timestamp = 0
    bar_started = False
    
    last_processed_idx = start_idx
    
    for i in range(start_idx, n):
        price = prices[i]
        volume = volumes[i]
        timestamp = timestamps[i]
        
        # 유효성 검사
        if np.isnan(price) or np.isnan(volume) or price <= 0 or volume < 0:
            continue
        
        dollar_amount = price * volume
        
        # 첫 번째 틱이거나 새 바 시작
        if not bar_started:
            current_open = price
            current_high = price
            current_low = price
            current_timestamp = timestamp
            bar_started = True
        else:
            # 고가/저가 업데이트
            if price > current_high:
                current_high = price
            if price < current_low:
                current_low = price
        
        current_close = price
        current_dollar_volume += dollar_amount
        last_processed_idx = i
        
        # 임계값 도달 시 바 완성
        if current_dollar_volume >= dollar_threshold:
            bar_timestamps.append(current_timestamp)
            bar_opens.append(current_open)
            bar_highs.append(current_high)
            bar_lows.append(current_low)
            bar_closes.append(current_close)
            
            # 다음 바를 위한 초기화
            current_dollar_volume = 0.0
            bar_started = False
    
    # numpy 배열로 변환
    result_timestamps = np.array(bar_timestamps, dtype=np.int64)
    result_opens = np.array(bar_opens, dtype=np.float64)
    result_highs = np.array(bar_highs, dtype=np.float64)
    result_lows = np.array(bar_lows, dtype=np.float64)
    result_closes = np.array(bar_closes, dtype=np.float64)
    
    return (result_timestamps, result_opens, result_highs, result_lows, 
            result_closes, current_dollar_volume, last_processed_idx)

def load_files_from_chunk_safe(chunk_files: List[str]) -> List[pd.DataFrame]:
    """
    🔥 ThreadPoolExecutor를 사용한 안전한 파일 로딩
    """
    chunk_dfs = []
    
    def read_file_thread_safe(file_path: str) -> Optional[pd.DataFrame]:
        try:
            df = pd.read_parquet(file_path)
            if df is not None and not df.empty:
                return df
            return None
        except Exception as e:
            tqdm.write(f"❌ 파일 로드 실패: {file_path}, 에러: {e}")
            return None
    
    with ThreadPoolExecutor(max_workers=min(8, len(chunk_files))) as executor:
        futures = [executor.submit(read_file_thread_safe, f) for f in chunk_files]
        
        for future in tqdm(as_completed(futures), total=len(futures), 
                          desc="📁 파일 로딩", leave=False):
            try:
                result = future.result(timeout=60)
                if result is not None:
                    chunk_dfs.append(result)
            except Exception as e:
                tqdm.write(f"⚠️ 스레드 에러: {e}")
    
    return chunk_dfs

def prepare_data_with_column_mapping(combined_df: pd.DataFrame, column_mapping: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    🔧 컬럼 매핑을 사용해서 데이터 준비
    """
    # 필수 컬럼 확인 및 매핑
    timestamp_col = column_mapping.get('timestamp')
    price_col = column_mapping.get('price')
    volume_col = column_mapping.get('volume')
    
    # timestamp 처리
    if timestamp_col and timestamp_col in combined_df.columns:
        timestamps = combined_df[timestamp_col].values
    else:
        tqdm.write("⚠️ timestamp 컬럼을 찾을 수 없어서 인덱스를 사용합니다")
        timestamps = np.arange(len(combined_df), dtype=np.int64)
    
    # price 처리
    if price_col and price_col in combined_df.columns:
        prices = combined_df[price_col].values.astype(np.float64)
    else:
        raise ValueError(f"❌ price 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {combined_df.columns.tolist()}")
    
    # volume 처리
    if volume_col and volume_col in combined_df.columns:
        volumes = combined_df[volume_col].values.astype(np.float64)
        tqdm.write(f"✅ volume 컬럼 '{volume_col}' 사용")
    else:
        # volume이 없으면 기본값 1.0 사용
        tqdm.write("⚠️ volume 컬럼을 찾을 수 없어서 기본값 1.0을 사용합니다")
        volumes = np.ones(len(combined_df), dtype=np.float64)
    
    return timestamps, prices, volumes


def process_chunk_with_state_carryover(
    chunk_files: List[str], 
    leftover_df: Optional[pd.DataFrame] = None,
    accumulated_dollar_volume: float = 0.0,
    dollar_threshold:float = 152936804.0,
    column_mapping: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, float, dict]:
    """
    🔥 상태 유지가 적용된 청크 처리 함수 (컬럼 매핑 지원)
    """
    
    # 1단계: 안전한 파일 로딩
    chunk_dfs = load_files_from_chunk_safe(chunk_files)
    
    if not chunk_dfs:
        return pd.DataFrame(), leftover_df, accumulated_dollar_volume, column_mapping
    
    # 2단계: 데이터 병합
    chunk_trades_df = pd.concat(chunk_dfs, ignore_index=True)
    
    # 3단계: 상태 유지 - 이전 청크의 찌꺼기와 현재 청크 합치기
    if leftover_df is not None and not leftover_df.empty:
        tqdm.write(f"🔄 이전 청크 찌꺼기 {len(leftover_df)}개 데이터 연결")
        combined_df = pd.concat([leftover_df, chunk_trades_df], ignore_index=True)
    else:
        combined_df = chunk_trades_df
    
    # 4단계: 컬럼 매핑 (첫 번째 청크에서만)
    if column_mapping is None:
        column_mapping = detect_column_mappings(combined_df)
    
    # 5단계: 데이터 정렬 및 준비
    timestamp_col = column_mapping.get('timestamp')
    if timestamp_col and timestamp_col in combined_df.columns:
        combined_df = combined_df.sort_values(timestamp_col).reset_index(drop=True)
    
    # 6단계: 데이터 추출 및 변환
    timestamps, prices, volumes = prepare_data_with_column_mapping(combined_df, column_mapping)
    
    # 7단계: Numba 최적화된 달러 바 생성
    (bar_timestamps, bar_opens, bar_highs, bar_lows, bar_closes, 
     remaining_dollar_volume, last_processed_idx) = create_dollar_bars_numba_safe(
        timestamps, prices, volumes, dollar_threshold, accumulated_dollar_volume
    )
    
    # 8단계: 완성된 바 데이터프레임 생성
    new_bars_df = pd.DataFrame({
        'timestamp': bar_timestamps,
        'open': bar_opens,
        'high': bar_highs,
        'low': bar_lows,
        'close': bar_closes
    })
    
    # 9단계: 다음 청크를 위한 찌꺼기 데이터 생성
    if last_processed_idx < len(combined_df) - 1:
        leftover_for_next = combined_df.iloc[last_processed_idx + 1:].copy()
        tqdm.write(f"📦 다음 청크로 넘길 찌꺼기: {len(leftover_for_next)}개 데이터")
    else:
        leftover_for_next = pd.DataFrame()
    
    return new_bars_df, leftover_for_next, remaining_dollar_volume, column_mapping

def process_all_files_with_state_management(
    file_paths: List[str],
    chunk_size: int = 20,
    dollar_threshold: float =152936804.0,
    output_file: str = "dollar_bars_result.parquet"
) -> pd.DataFrame:
    """
    🔥 전체 파일 처리 메인 함수 - 완전 안전한 버전
    """
    
    # 파일을 청크로 분할
    file_chunks = [file_paths[i:i + chunk_size] for i in range(0, len(file_paths), chunk_size)]
    num_chunks = len(file_chunks)
    
    # 전체 결과 저장용
    all_bars = []
    
    # 상태 유지 변수들
    leftover_df = None
    accumulated_dollar_volume = 0.0
    column_mapping = None
    
    tqdm.write(f"🚀 총 {len(file_paths)}개 파일을 {num_chunks}개 청크로 처리 시작!")
    tqdm.write(f"💰 달러 바 임계값: ${dollar_threshold:,.0f}")
    
    with tqdm(total=num_chunks, desc="🏗️ 청크 처리 진행도", unit="청크") as pbar:
        for i, chunk in enumerate(file_chunks):
            pbar.set_postfix_str(f"청크 {i+1}/{num_chunks} | 파일 {len(chunk)}개")
            
            try:
                # 청크 처리 (상태 유지 적용)
                new_bars_df, leftover_df, accumulated_dollar_volume, column_mapping = process_chunk_with_state_carryover(
                    chunk, leftover_df, accumulated_dollar_volume, dollar_threshold, column_mapping
                )
                
                # 완성된 바가 있으면 결과에 추가
                if not new_bars_df.empty:
                    all_bars.append(new_bars_df)
                    tqdm.write(f"✅ 청크 {i+1}: {len(new_bars_df)}개 달러 바 생성 완료")
                else:
                    tqdm.write(f"⚠️ 청크 {i+1}: 생성된 달러 바 없음")
                    
            except Exception as e:
                tqdm.write(f"❌ 청크 {i+1} 처리 실패: {e}")
                continue
            
            pbar.update(1)
    
    # 모든 결과 병합
    if all_bars:
        final_result = pd.concat(all_bars, ignore_index=True)
        final_result = final_result.sort_values('timestamp').reset_index(drop=True)
        
        # 결과 저장
        final_result.to_parquet(output_file, index=False)
        tqdm.write(f"🎉 최종 결과: {len(final_result)}개 달러 바 생성 완료!")
        tqdm.write(f"💾 결과 저장됨: {output_file}")
        
        return final_result
    else:
        tqdm.write("⚠️ 생성된 달러 바가 없습니다.")
        return pd.DataFrame()

# 🔥 실제 사용 예시
if __name__ == "__main__":
    # 파일 경로 설정
    data_folder = Path("C:\Monilusion\daily_cache")
    all_files = [str(f) for f in data_folder.glob("*.parquet")]
    
    # 파라미터 설정
    CHUNK_SIZE = 20
    DOLLAR_THRESHOLD = 152936804.0
    
    print(f"🎯 발견된 파일 개수: {len(all_files)}개")
    
    # 처리 실행
    result_df = process_all_files_with_state_management(
        all_files, 
        chunk_size=CHUNK_SIZE,
        dollar_threshold=DOLLAR_THRESHOLD,
        output_file="btc_dollar_bars_optimized.parquet"
    )
    
    if not result_df.empty:
        print(f"\n🏆 최종 성H과:")
        print(f"   📈 생성된 달러 바: {len(result_df):,}개")
        print(f"   ⏰ 시간 범위: {pd.to_datetime(result_df['timestamp'].min(), unit='ms')} ~ {pd.to_datetime(result_df['timestamp'].max(), unit='ms')}")
        print(f"   💰 가격 범위: ${result_df['low'].min():.2f} ~ ${result_df['high'].max():.2f}")
