import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import urllib.parse

# --- CẤU HÌNH CHUNG ---
BASE_URL     = 'https://your-domain.com/'   # Thay bằng domain thực tế
OUTPUT_CSV   = 'svbk_all_combinations.csv'
REQUEST_DELAY = 0.5  # giây nghỉ giữa các request

# --- HÀM LẤY OPTIONS TỪ SELECT ---
def get_select_options():
    """
    Truy vấn trang chính, parse <select id="khoa"> và <select id="truong">,
    trả về dict: {'khoa': [(value, label), ...], 'truong': [...]}
    """
    resp = requests.get(BASE_URL)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')
    opts = {}

    for select_id in ('khoa', 'truong'):
        sel = soup.find('select', id=select_id)
        if not sel:
            raise RuntimeError(f"Không tìm thấy <select id=\"{select_id}\"> trên trang.")
        items = []
        for option in sel.find_all('option'):
            val = option.get('value', '').strip()
            label = option.get_text(strip=True)
            # Bỏ mục trống hoặc placeholder
            if val and label and not label.lower().startswith('--'):
                items.append((val, label))
        opts[select_id] = items

    return opts

# --- HÀM LẤY DỮ LIỆU TRONG 1 TRANG ---
def fetch_page(page, khoa, truong):
    """
    Trả về list các dict cho page hiện tại,
    với filter params khoa, truong.
    """
    params = {
        'khoa': khoa,
        'truong': truong,
        'page': page
    }
    url = BASE_URL + '?' + urllib.parse.urlencode(params, safe='áéàÂ...')
    resp = requests.get(url)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, 'html.parser')

    table = soup.find('table', class_='w-full')
    if not table or not table.tbody:
        return []
    data = []
    for tr in table.tbody.find_all('tr'):
        cols = tr.find_all('td')
        if len(cols) < 5:
            continue
        data.append({
            'rank': cols[0].get_text(strip=True),
            'mssv': cols[1].get_text(strip=True),
            'name': cols[2].get_text(strip=True),
            'cpa':  cols[3].get_text(strip=True),
            'lop':  cols[4].get_text(strip=True),
        })
    return data

# --- HÀM CRAWL TOÀN BỘ TRANG CHO 1 TỔ HỢP ---
def crawl_for(khoa, truong):
    """
    Lặp fetch_page từ page=1 lên cho đến khi fetch_page trả về [].
    Kết quả gắn thêm khóa và trường.
    """
    records = []
    page = 1
    while True:
        print(f'  → Crawling khoa={khoa}, truong={truong}, page={page} …')
        rows = fetch_page(page, khoa, truong)
        if not rows:
            break
        for r in rows:
            r['khoa'] = khoa
            r['truong'] = truong
        records.extend(rows)
        page += 1
        time.sleep(REQUEST_DELAY)
    return records

# --- MAIN ---
if __name__ == '__main__':
    print('1) Lấy danh sách các Khóa & Trường/Viện…')
    opts = get_select_options()

    all_data = []
    for khoa_val, khoa_label in opts['khoa']:
        for truong_val, truong_label in opts['truong']:
            # bạn có thể bỏ if sau nếu muốn cả tổ hợp kể cả khi không có dữ liệu
            data = crawl_for(khoa_val, truong_val)
            if data:
                all_data.extend(data)

    # Chuyển sang DataFrame và lưu CSV
    df = pd.DataFrame(all_data)
    print(f'Tổng bản ghi thu thập: {len(df)}')
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f'Đã lưu vào {OUTPUT_CSV}')
