import requests
from bs4 import BeautifulSoup
import json
import time
import re

# ==================================================
# 1. 基本設定（測試版）
# ==================================================

BASE_URL = "https://wear.jp"
START_PAGE = 1
END_PAGE = 80
MAX_OUTFITS_PER_PAGE = 50

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) OutfitResearchBot/1.0",
    "Accept": "text/html"
}

OUTPUT_FILE = "wear_outfit_pairs_color_separated_test.json"

# ==================================================
# 2. 上衣 / 下裝 類型
# ==================================================

TOP_KEYWORDS = {
    "t-shirt": ["tシャツ", "t-shirt", "tee"],
    "shirt": ["シャツ", "shirt"],
    "hoodie": ["パーカー", "hoodie"],
    "sweater": ["ニット", "sweater", "セーター"],
    "blouse": ["ブラウス", "blouse"],
    "jacket": ["ジャケット", "jacket"],
    "coat": ["コート", "coat"],
    "cardigan": ["カーディガン", "cardigan"],
    "vest": ["ベスト", "vest"],
    "tank top": ["タンクトップ", "tank top"],
    "camisole": ["キャミソール", "camisole"]
}

BOTTOM_KEYWORDS = {
    "jeans": ["デニム", "jeans", "ジーンズ"],
    "wide pants": ["ワイドパンツ", "ワイド"],
    "slim pants": ["スリムパンツ", "スリム"],
    "flare pants": ["フレアパンツ", "フレア"],
    "pants": ["パンツ"],
    "skirt": ["スカート", "skirt"],
    "shorts": ["ショートパンツ", "shorts"],
    "capris": ["七分丈", "capris"],
    "leggings": ["レギンス", "leggings"],
    "sweat pants": ["スウェットパンツ", "sweat pants"]
}

# ==================================================
# 3. 細顏色（日文 → 英文）
# ==================================================

JP_COLORS_FINE = {
    "black": ["黒", "ブラック"],
    "white": ["白", "ホワイト"],
    "gray": ["グレー", "灰色"],
    "light blue": ["ライトブルー"],
    "blue": ["青", "ブルー"],
    "navy": ["ネイビー", "紺"],
    "beige": ["ベージュ"],
    "brown": ["ブラウン", "茶"],
    "green": ["グリーン", "緑"],
    "red": ["レッド", "赤", "紅"],
    "pink": ["ピンク", "桃色"],
    "orange": ["オレンジ"],
    "yellow": ["イエロー", "黄色"],
    "purple": ["紫", "パープル"],
    "gold": ["ゴールド", "金"],
    "silver": ["シルバー", "銀"],
    "cream": ["クリーム"],
    "khaki": ["カーキ"]
}

# ==================================================
# 4. 基礎抽取工具
# ==================================================

def extract_gender(text):
    if any(k in text for k in ["レディース", "women", "女性"]):
        return "female"
    if any(k in text for k in ["メンズ", "men", "男性"]):
        return "male"
    return None


def find_positions(text, keywords):
    positions = []
    for kw in keywords:
        for m in re.finditer(kw, text):
            positions.append(m.start())
    return positions


def find_color_positions(text):
    results = []
    for color, kws in JP_COLORS_FINE.items():
        for kw in kws:
            for m in re.finditer(kw, text):
                results.append((color, m.start()))
    return results

# ==================================================
# 5. 顏色分配（核心！）
# ==================================================

def assign_color(item_positions, color_positions):
    if not item_positions or not color_positions:
        return None

    min_dist = float("inf")
    chosen_color = None

    for item_pos in item_positions:
        for color, color_pos in color_positions:
            dist = abs(item_pos - color_pos)
            if dist < min_dist:
                min_dist = dist
                chosen_color = color

    return chosen_color

# ==================================================
# 6. 抓單一穿搭頁
# ==================================================

def crawl_outfit_page(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=8)
    except requests.exceptions.RequestException:
        return None, "request_error"

    if r.status_code != 200:
        return None, "http_error"

    soup = BeautifulSoup(r.text, "html.parser")
    text = soup.get_text(" ", strip=True).lower()

    gender = extract_gender(text)

    top_type = None
    bottom_type = None
    top_positions = []
    bottom_positions = []

    # Find all matching tops and pick the one with most matches
    top_matches = {}
    for t, kws in TOP_KEYWORDS.items():
        pos = find_positions(text, kws)
        if pos:
            top_matches[t] = pos

    if top_matches:
        top_type = max(top_matches, key=lambda k: len(top_matches[k]))
        top_positions = top_matches[top_type]

    # Find all matching bottoms and pick the one with most matches
    bottom_matches = {}
    for b, kws in BOTTOM_KEYWORDS.items():
        pos = find_positions(text, kws)
        if pos:
            bottom_matches[b] = pos

    if bottom_matches:
        bottom_type = max(bottom_matches, key=lambda k: len(bottom_matches[k]))
        bottom_positions = bottom_matches[bottom_type]

    color_positions = find_color_positions(text)

    top_color = assign_color(top_positions, color_positions)
    bottom_color = assign_color(bottom_positions, color_positions)

    if not (gender and top_type and bottom_type):
        return None, "missing_fields"

    if top_color is None or bottom_color is None:
        return None, "missing_color"

    return {
        "gender": gender,
        "top": {
            "type": top_type,
            "color": top_color
        },
        "bottom": {
            "type": bottom_type,
            "color": bottom_color
        }
    }, "ok"

# ==================================================
# 7. 主爬蟲（五欄位完全相同才去重）
# ==================================================

def crawl_wear():
    outfits = []
    seen_urls = set()
    stats = {
        "links": 0,
        "duplicate_url": 0,
        "ok": 0,
        "missing_fields": 0,
        "missing_color": 0,
        "http_error": 0,
        "request_error": 0,
    }

    for page in range(START_PAGE, END_PAGE + 1):
        print(f"\n→ Crawling WEAR page {page}")
        r = requests.get(f"{BASE_URL}/coordinate/?pageno={page}", headers=HEADERS)
        soup = BeautifulSoup(r.text, "html.parser")

        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/") and href.count("/") >= 2:
                if any(x in href for x in ["shop", "search", "tag"]):
                    continue
                links.append(BASE_URL + href)

        links = list(dict.fromkeys(links))[:MAX_OUTFITS_PER_PAGE]
        stats["links"] += len(links)
        print(f"  Found {len(links)} outfits")

        for i, url in enumerate(links, 1):
            if url in seen_urls:
                stats["duplicate_url"] += 1
                continue
            seen_urls.add(url)

            print(f"    [{i}/{len(links)}] parsing")
            outfit, status = crawl_outfit_page(url)

            if status != "ok":
                stats[status] = stats.get(status, 0) + 1
                continue

            stats["ok"] += 1
            outfits.append(outfit)
            time.sleep(0.5)

    print("\nStats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    return outfits

# ==================================================
# 8. 執行
# ==================================================

def main():
    data = crawl_wear()
    print(f"\nCollected {len(data)} unique outfit pairs")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Saved:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
