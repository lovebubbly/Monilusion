# revenge_downloader_unicode_fixed.py - 유니코드 완전 정복 버전!
import sys
import io

# 🔥 유니코드 마스터 설정 (맨 위에 배치!)
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

import pandas as pd
import asyncio
import aiohttp
from aiohttp import TCPConnector, ClientTimeout
import zipfile
import io
import os
from tqdm import tqdm
from datetime import datetime, timedelta
import time
import random
import logging
from typing import List, Set
import json

# 기존 설정들...
SYMBOL = 'BTCUSDT'
START_DATE_STR = '2021-01-01'
END_DATE_STR = '2025-06-01'
DOLLAR_BAR_THRESHOLD = 5_000_000
OUTPUT_DIR = 'data'
TEMP_DIR = 'daily_cache'

CONCURRENT_DOWNLOADS = 15
MAX_RETRIES = 8
BASE_RETRY_DELAY = 3
MAX_RETRY_DELAY = 120
BATCH_SIZE = 100
PROGRESS_FILE = 'smart_download_progress.json'

# 🎯 유니코드 안전 로깅 설정
class UnicodeFormatter(logging.Formatter):
    """유니코드 안전 로거 포매터"""
    def format(self, record):
        try:
            # 기본 포매팅
            msg = super().format(record)
            # UTF-8으로 안전하게 인코딩
            return msg.encode('utf-8', errors='replace').decode('utf-8')
        except Exception:
            # 최악의 경우 아스키만 남기기
            return str(record.getMessage()).encode('ascii', errors='replace').decode('ascii')

# 로깅 설정 (유니코드 완전 대응)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('download.log', encoding='utf-8'),  # 파일도 UTF-8!
        logging.StreamHandler(sys.stdout)  # 이미 UTF-8로 설정된 stdout 사용
    ]
)

# 모든 핸들러에 유니코드 포매터 적용
for handler in logging.getLogger().handlers:
    handler.setFormatter(UnicodeFormatter())

logger = logging.getLogger(__name__)

def exponential_backoff_delay(attempt: int, base_delay: float = BASE_RETRY_DELAY) -> float:
    """지수 백오프 계산 (지터 포함)"""
    delay = min(base_delay * (2 ** attempt), MAX_RETRY_DELAY)
    jitter = random.uniform(0.1, 0.3) * delay
    return delay + jitter

def save_smart_progress(completed_dates: Set[str], failed_dates: Set[str]):
    """지능형 진행률 저장 (UTF-8 강제)"""
    progress = {
        'completed_dates': list(completed_dates),
        'failed_dates': list(failed_dates),
        'timestamp': datetime.now().isoformat(),
        'total_completed': len(completed_dates),
        'total_failed': len(failed_dates)
    }
    with open(PROGRESS_FILE, 'w', encoding='utf-8') as f:  # UTF-8 명시!
        json.dump(progress, f, indent=2, ensure_ascii=False)

def load_smart_progress():
    """지능형 진행률 로드"""
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:  # UTF-8 명시!
                data = json.load(f)
                return set(data.get('completed_dates', [])), set(data.get('failed_dates', []))
        except Exception as e:
            logger.warning(f"진행률 파일 로드 실패: {e}, 처음부터 시작")
    return set(), set()

# 🎨 이모지 안전 로그 함수들
def safe_log_success(date_str: str, count: int, attempt: int):
    """안전한 성공 로그"""
    try:
        logger.info(f"✅ {date_str}: {count:,}건 저장 완료 (시도 {attempt})")
    except UnicodeEncodeError:
        logger.info(f"[SUCCESS] {date_str}: {count:,}건 저장 완료 (시도 {attempt})")

def safe_log_error(date_str: str, message: str):
    """안전한 에러 로그"""
    try:
        logger.error(f"❌ {date_str}: {message}")
    except UnicodeEncodeError:
        logger.error(f"[ERROR] {date_str}: {message}")

def safe_log_warning(date_str: str, message: str):
    """안전한 경고 로그"""
    try:
        logger.warning(f"⚠️ {date_str}: {message}")
    except UnicodeEncodeError:
        logger.warning(f"[WARNING] {date_str}: {message}")

async def smart_download_with_backoff_unicode_safe(session, date_str, symbol, temp_dir, completed_dates, failed_dates):
    """유니코드 안전 버전 다운로드 함수"""
    cache_file = f"{temp_dir}/{date_str}.parquet"
    
    if date_str in completed_dates or os.path.exists(cache_file):
        return 'already_completed'
    
    if date_str in failed_dates:
        return 'previously_failed'
    
    url = f"https://data.binance.vision/data/futures/um/daily/trades/{symbol}/{symbol}-trades-{date_str}.zip"
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if attempt > 1:
                delay = exponential_backoff_delay(attempt - 1)
                try:
                    logger.info(f"🔄 {date_str}: {delay:.1f}초 대기 후 {attempt}번째 시도")
                except UnicodeEncodeError:
                    logger.info(f"[RETRY] {date_str}: {delay:.1f}초 대기 후 {attempt}번째 시도")
                await asyncio.sleep(delay)
            else:
                await asyncio.sleep(random.uniform(0.1, 1.0))
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                if response.status == 200:
                    content = await response.read()
                    
                    with zipfile.ZipFile(io.BytesIO(content)) as z:
                        csv_filename = z.namelist()[0]
                        with z.open(csv_filename) as csv_file:
                            df = pd.read_csv(
                                csv_file,
                                header=None,
                                dtype='str',
                                low_memory=False
                            )
                            
                            if len(df.columns) >= 6:
                                df.columns = ['trade_id', 'price', 'qty', 'quote_qty', 'timestamp', 'is_buyer_maker']
                                
                                try:
                                    df['trade_id'] = pd.to_numeric(df['trade_id'], errors='coerce')
                                    df['price'] = pd.to_numeric(df['price'], errors='coerce')
                                    df['qty'] = pd.to_numeric(df['qty'], errors='coerce')
                                    df['quote_qty'] = pd.to_numeric(df['quote_qty'], errors='coerce')
                                    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                                    df['is_buyer_maker'] = df['is_buyer_maker'].map({
                                        'true': True, 'false': False, True: True, False: False
                                    })
                                    
                                    df = df.dropna()
                                    
                                    if len(df) > 0:
                                        df.to_parquet(cache_file, compression='snappy')
                                        completed_dates.add(date_str)
                                        safe_log_success(date_str, len(df), attempt)
                                        return 'success'
                                        
                                except Exception as e:
                                    safe_log_error(date_str, f"데이터 처리 실패 - {e}")
                                    failed_dates.add(date_str)
                                    return 'data_error'
                            else:
                                safe_log_error(date_str, f"컬럼 부족 ({len(df.columns)}개)")
                                failed_dates.add(date_str)
                                return 'format_error'
                                
                elif response.status == 404:
                    safe_log_warning(date_str, "데이터 없음 (404)")
                    return 'not_found'
                elif response.status == 429:
                    safe_log_warning(date_str, "Rate limit (429) - 더 오래 대기")
                    await asyncio.sleep(exponential_backoff_delay(attempt + 2))
                    continue
                elif response.status >= 500:
                    safe_log_warning(date_str, f"서버 오류 ({response.status}) - 재시도")
                    continue
                else:
                    safe_log_warning(date_str, f"HTTP {response.status}")
                    
        except asyncio.TimeoutError:
            safe_log_warning(date_str, f"타임아웃 (시도 {attempt})")
        except aiohttp.ClientConnectorError as e:
            safe_log_warning(date_str, f"연결 오류 (시도 {attempt}) - {e}")
        except Exception as e:
            safe_log_warning(date_str, f"예상치 못한 오류 (시도 {attempt}) - {e}")
    
    safe_log_error(date_str, "최대 재시도 횟수 초과")
    failed_dates.add(date_str)
    return 'max_retries_exceeded'

async def smart_revenge_download_unicode_master():
    """유니코드 마스터 다운로드 시스템"""
    try:
        # 🎨 안전한 이모지 출력
        print("🧠🔥 지능형 복수 다운로더 시작! (유니코드 마스터 버전) 🔥🧠")
    except UnicodeEncodeError:
        print("[UNICODE MASTER] 지능형 복수 다운로더 시작!")
    
    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 날짜 리스트 생성
    start_date = datetime.strptime(START_DATE_STR, '%Y-%m-%d').date()
    end_date = datetime.strptime(END_DATE_STR, '%Y-%m-%d').date()
    dates = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') 
             for i in range((end_date - start_date).days)]
    
    # 지능형 진행률 로드
    completed_dates, failed_dates = load_smart_progress()
    remaining_dates = [d for d in dates if d not in completed_dates and d not in failed_dates]
    
    # 안전한 출력
    print(f"📊 전체: {len(dates)}일")
    print(f"✅ 완료: {len(completed_dates)}일")
    print(f"❌ 실패: {len(failed_dates)}일")
    print(f"⏳ 남은 작업: {len(remaining_dates)}일")
    print(f"🎯 동시 연결: {CONCURRENT_DOWNLOADS}개")
    
    if len(remaining_dates) == 0:
        if len(failed_dates) > 0:
            retry_failed = input(f"\n❓ {len(failed_dates)}개 실패 파일 재시도? (y/n): ")
            if retry_failed.lower() == 'y':
                remaining_dates = list(failed_dates)
                failed_dates.clear()
            else:
                print("🎉 작업 완료!")
                return
        else:
            print("🎉 모든 다운로드 완료!")
            return
    
    # 배치 단위로 처리
    for batch_start in range(0, len(remaining_dates), BATCH_SIZE):
        batch_dates = remaining_dates[batch_start:batch_start + BATCH_SIZE]
        batch_num = batch_start // BATCH_SIZE + 1
        total_batches = (len(remaining_dates) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\n🔄 배치 {batch_num}/{total_batches} 처리 중... ({len(batch_dates)}개)")
        
        semaphore = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
        connector = TCPConnector(
            limit=CONCURRENT_DOWNLOADS * 2,
            limit_per_host=CONCURRENT_DOWNLOADS,
            ttl_dns_cache=300,
            use_dns_cache=True,
            keepalive_timeout=30
        )
        timeout = ClientTimeout(total=60, connect=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            async def download_with_semaphore(date_str):
                async with semaphore:
                    return await smart_download_with_backoff_unicode_safe(
                        session, date_str, SYMBOL, TEMP_DIR, completed_dates, failed_dates
                    )
            
            tasks = [download_with_semaphore(date) for date in batch_dates]
            
            start_time = time.time()
            results = []
            
            # tqdm도 유니코드 안전하게
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), 
                         desc=f"🧠 배치 {batch_num} 유니코드 마스터 다운로드", 
                         ascii=True):  # ASCII 모드로 안전하게!
                result = await f
                results.append(result)
            
            # 배치 완료 후 진행률 저장
            save_smart_progress(completed_dates, failed_dates)
            
            elapsed = time.time() - start_time
            success_count = results.count('success')
            already_completed = results.count('already_completed')
            
            print(f"⚡ 배치 {batch_num} 완료!")
            print(f"🎯 성공: {success_count}개")
            print(f"✅ 이미 완료: {already_completed}개") 
            print(f"❌ 실패: {len([r for r in results if r not in ['success', 'already_completed']])}개")
            print(f"⏱️ 소요시간: {elapsed:.1f}초")
            
            if batch_num < total_batches:
                rest_time = random.uniform(5, 15)
                print(f"😴 배치 간 휴식: {rest_time:.1f}초")
                await asyncio.sleep(rest_time)
    
    # 최종 통계
    final_completed, final_failed = load_smart_progress()
    total_success_rate = (len(final_completed) / len(dates)) * 100
    
    print(f"\n🎉 유니코드 마스터 다운로드 완료!")
    print(f"📊 최종 성공률: {total_success_rate:.1f}%")
    print(f"✅ 성공: {len(final_completed)}개")
    print(f"❌ 실패: {len(final_failed)}개")
    
    return final_completed, final_failed

if __name__ == '__main__':
    asyncio.run(smart_revenge_download_unicode_master())
