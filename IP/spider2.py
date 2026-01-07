import requests
import json
import time
import re

# ==================================================
# 1. 基本設定
# ==================================================

SUBREDDITS = {
    "male": [
        "malefashionadvice",
        "streetwear"
    ],
    "female": [
        "femalefashionadvice",
        "OUTFITS"
    ]
}

POST_LIMIT = 2000         # 每個 subreddit 最多抓幾篇
SLEEP = 1.0               # Reddit 請求間隔（不要太快）

HEADERS = {
    "User-Agent": "OutfitResearchBot/1.0 (academic project)"
}

OUTPUT_FILE = "reddit_outfit_pairs.json"

# ==================================================
# 2. 類型與顏色字典（英文）
# ==================================================

TOP_KEYWORDS = {
    "t-shirt": ["t-shirt", "tee", "tshirt"],
    "shirt": ["shirt", "button up", "button-up"],
    "hoodie": ["hoodie", "sweatshirt"],
    "sweater": ["sweater", "knit", "pullover", "jumper"],
    "jacket": ["jacket", "blazer"],
    "coat": ["coat", "overcoat", "trench"],
    "cardigan": ["cardigan"],
    "top": ["top"]
}

BOTTOM_KEYWORDS = {
    "jeans": ["jeans", "denim"],
    "wide pants": ["wide pants", "wide trousers", "wide leg"],
    "pants": ["pants", "trousers", "slacks", "chinos"],
    "shorts": ["shorts"],
    "skirt": ["skirt"],
    "leggings": ["leggings"],
    "joggers": ["joggers", "sweatpants"]
}

COLORS = {
    "black": ["black"],
    "white": ["white", "cream"],
    "gray": ["gray", "grey", "charcoal"],
    "navy": ["navy"],
    "blue": ["blue"],
    "light blue": ["light blue", "baby blue"],
    "dark blue": ["dark blue"],
    "beige": ["beige", "tan", "khaki"],
    "brown": ["brown", "chocolate"],
    "green": ["green"],
    "olive": ["olive"],
    "red": ["red", "burgundy", "maroon"],
    "pink": ["pink"],
    "yellow": ["yellow"],
    "orange": ["orange"],
    "purple": ["purple"]
}

# ==================================================
# 3. 抽取工具
# ==================================================

def extract_category(text, mapping):
    for cat, kws in mapping.items():
        for kw in kws:
            if kw in text:
                return cat
    return None


def extract_all_colors(text):
    found = []
    for color, kws in COLORS.items():
        for kw in kws:
            if kw in text:
                found.append(color)
                break
    return list(dict.fromkeys(found))


# ==================================================
# 4. 解析一篇 Reddit 貼文
# ==================================================

def parse_post(text, gender):
    text = text.lower()

    top_type = extract_category(text, TOP_KEYWORDS)
    bottom_type = extract_category(text, BOTTOM_KEYWORDS)
    colors = extract_all_colors(text)

    # 必須有上下裝，但顏色可以用預設值
    if not top_type or not bottom_type:
        return None

    # 顏色分配策略
    if len(colors) == 0:
        # 沒有顏色資訊，用預設值
        top_color = "black" if gender == "male" else "white"
        bottom_color = "black"
    elif len(colors) == 1:
        top_color = bottom_color = colors[0]
    else:
        top_color = colors[0]
        bottom_color = colors[1]

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
    }

# ==================================================
# 5. 主爬蟲（支援 Pagination）
# ==================================================

def crawl_reddit():
    outfits = []
    seen = set()

    for gender, subs in SUBREDDITS.items():
        for sub in subs:
            print(f"\n→ Crawling r/{sub}", flush=True)
            
            total_fetched = 0
            after = None
            valid_outfits = 0
            
            while total_fetched < POST_LIMIT:
                # 構建 URL（每次最多 100）
                batch_size = min(100, POST_LIMIT - total_fetched)
                url = f"https://www.reddit.com/r/{sub}/top.json?t=all&limit={batch_size}"
                if after:
                    url += f"&after={after}"
                
                r = requests.get(url, headers=HEADERS)
                
                if r.status_code != 200:
                    print(f"  ✖ Failed (status {r.status_code})", flush=True)
                    break
                
                json_data = r.json()
                posts = json_data["data"]["children"]
                after = json_data["data"].get("after")
                
                if not posts:
                    print(f"  ⚠ No more posts available", flush=True)
                    break
                
                total_fetched += len(posts)
                
                for post in posts:
                    data = post["data"]
                    text = (data.get("title", "") + " " + data.get("selftext", "")).strip()
                    
                    outfit = parse_post(text, gender)
                    if not outfit:
                        continue
                    
                    key = (
                        outfit["gender"],
                        outfit["top"]["type"],
                        outfit["top"]["color"],
                        outfit["bottom"]["type"],
                        outfit["bottom"]["color"]
                    )
                    
                    if key in seen:
                        continue
                    
                    seen.add(key)
                    outfits.append(outfit)
                    valid_outfits += 1
                
                print(f"  Progress: {total_fetched} posts fetched, {valid_outfits} valid outfits", flush=True)
                
                # 如果沒有下一頁，停止
                if not after:
                    print(f"  ✓ Reached end of subreddit", flush=True)
                    break
                
                time.sleep(SLEEP)
            
            print(f"  ✓ Finished r/{sub}: {valid_outfits} outfits collected", flush=True)

    return outfits

# ==================================================
# 6. 執行
# ==================================================

def main():
    print("=" * 60, flush=True)
    print("🚀 開始爬取 Reddit 穿搭數據...", flush=True)
    print(f"目標: 每個 subreddit 最多 {POST_LIMIT} 篇", flush=True)
    print("=" * 60, flush=True)
    
    data = crawl_reddit()
    
    print("\n" + "=" * 60, flush=True)
    print(f"✅ 收集完成！共 {len(data)} 組獨特穿搭配對", flush=True)
    print("=" * 60, flush=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\n💾 已保存至: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
