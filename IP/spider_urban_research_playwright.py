import asyncio
import json
import re
import time
from pathlib import Path

from playwright.async_api import async_playwright

# =============================
# Config
# =============================

SEED_URL = "https://www.urban-research.tw/Form/Coordinate/CoordinateList.aspx"
MAX_PAGES = 20      # seed pages to iterate
MAX_OUTFITS = 400  # total outfits to crawl
PAGE_WAIT = 1.5    # seconds to wait after load/scroll
SCROLL_ROUNDS = 3  # scroll times on seed page to load more cards
OUTPUT_FILE = Path("urban_research_outfit_pairs.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}

TOP_KEYWORDS = {
    "t-shirt": ["t恤", "tee", "上衣", "t-shirt"],
    "shirt": ["襯衫", "衬衫", "shirt"],
    "hoodie": ["連帽", "帽t", "hoodie", "連帽上衣"],
    "sweater": ["毛衣", "針織", "針織衫", "針織衣", "sweater"],
    "jacket": ["夾克", "外套", "風衣", "教練外套", "jacket"],
    "coat": ["大衣", "長版外套", "coat"],
    "cardigan": ["開襟", "cardigan", "罩衫"],
    "vest": ["背心", "vest"],
}

BOTTOM_KEYWORDS = {
    "dress": ["洋裝", "連身裙", "dress"],
    "jumpsuit": ["連身褲", "jumpsuit"],
    "romper": ["連身短褲", "romper"],
    "leggings": ["緊身褲", "瑜珈褲", "leggings"],
    "jeans": ["牛仔褲", "丹寧", "jeans"],
    "pants": ["褲", "褲子", "長褲", "西裝褲", "寬褲", "神褲", "pants", "工作褲"],
    "shorts": ["短褲", "短裤", "shorts"],
    "skirt": ["裙", "skirt"],
}

COLORS = {
    "black": ["黑", "黑色", "black"],
    "white": ["白", "白色", "white"],
    "gray": ["灰", "灰色", "灰白", "gray", "grey"],
    "blue": ["藍", "蓝", "藍色", "蓝色", "水洗藍", "淺藍", "深藍", "blue"],
    "navy": ["深藍", "藏青", "navy"],
    "beige": ["米色", "卡其", "米白", "膚", "beige"],
    "brown": ["棕", "棕色", "咖啡", "摩卡", "brown"],
    "green": ["綠", "绿", "綠色", "绿色", "軍綠", "green"],
    "red": ["紅", "红", "酒紅", "red"],
    "pink": ["粉", "粉色", "粉紅", "pink"],
    "yellow": ["黃", "黄", "黃色", "黄色", "yellow"],
    "orange": ["橙", "橙色", "橘", "橘色", "orange"],
    "purple": ["紫", "紫色", "purple"],
    "cream": ["奶油", "米白", "cream"],
}

# =============================
# Helpers
# =============================

def find_positions(text: str, keywords):
    pos = []
    for kw in keywords:
        for m in re.finditer(kw, text):
            pos.append(m.start())
    return pos


def find_color_positions(text: str):
    res = []
    for color, kws in COLORS.items():
        for kw in kws:
            for m in re.finditer(kw, text):
                res.append((color, m.start()))
    return res


def assign_color(item_positions, color_positions):
    if not item_positions or not color_positions:
        return None
    best = None
    best_d = 1e9
    for ip in item_positions:
        for c, cp in color_positions:
            d = abs(ip - cp)
            if d < best_d:
                best_d = d
                best = c
    return best


def extract_gender(text: str):
    # Always return female for this site
    return "female"


async def scroll_page(page):
    for _ in range(SCROLL_ROUNDS):
        await page.mouse.wheel(0, 2000)
        await page.wait_for_timeout(int(PAGE_WAIT * 1000))


async def extract_outfit_items(page):
    """Extract texts from product cards including brand, name and color information."""
    items = []

    # Get all product information sections - this should contain everything
    info_sections = await page.query_selector_all(".products__information")
    
    for section in info_sections:
        # Get all text from this section (includes brand, name, and color)
        text = await section.inner_text()
        if text:
            items.append(text)
    
    # Fallback: if no info sections found, try to extract from product cards
    if not items:
        product_cards = await page.query_selector_all("div.products__card")
        for card in product_cards:
            text = await card.inner_text()
            if text:
                items.append(text)
    
    # Last resort: get all text content from page
    if not items:
        body = await page.inner_text("body")
        if body:
            items.append(body)

    return [t.lower() for t in items if t]


def find_best_item(items, keyword_map):
    """Pick the item text with most hits for the given keyword map."""
    best_score = -1
    best_text = None
    best_type = None
    best_positions = []
    for text in items:
        matches = {}
        for k, kws in keyword_map.items():
            pos = find_positions(text, kws)
            if pos:
                matches[k] = pos
        if not matches:
            continue
        top_key = max(matches, key=lambda k: len(matches[k]))
        score = len(matches[top_key])
        if score > best_score or (score == best_score and matches[top_key][0] < (best_positions[0] if best_positions else 1e9)):
            best_score = score
            best_text = text
            best_type = top_key
            best_positions = matches[top_key]
    return best_text, best_type, best_positions


def parse_outfit(items):
    gender = extract_gender("")  # forced null

    top_text, top_type, top_pos = find_best_item(items, TOP_KEYWORDS)
    bottom_text, bottom_type, bottom_pos = find_best_item(items, BOTTOM_KEYWORDS)

    top_color = assign_color(top_pos, find_color_positions(top_text)) if top_text else None
    bottom_color = assign_color(bottom_pos, find_color_positions(bottom_text)) if bottom_text else None

    if not (top_type and bottom_type):
        return None
    if top_color is None or bottom_color is None:
        return None

    return {
        "gender": gender,
        "top": {"type": top_type, "color": top_color},
        "bottom": {"type": bottom_type, "color": bottom_color},
    }

# =============================
# Main crawl
# =============================

async def crawl():
    outfits = []
    seen_links = set()
    seen_data = set()

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # Set to False to debug
        page = await browser.new_page(extra_http_headers=HEADERS)

        # Collect coordinate detail links from staff coordinate pages
        print("Collecting staff coordinate links...")
        for page_no in range(1, MAX_PAGES + 1):
            seed = SEED_URL if page_no == 1 else f"{SEED_URL}?pno={page_no}"
            print(f"  Loading page {page_no}: {seed}")
            await page.goto(seed, wait_until="networkidle", timeout=30000)
            await page.wait_for_timeout(int(PAGE_WAIT * 1000))
            await scroll_page(page)
            
            # Find coordinate detail links
            before = len(seen_links)
            
            # Look for coordinate detail links
            coord_links = await page.query_selector_all("a[href*='CoordinateDetail']")
            for link in coord_links:
                href = await link.get_attribute("href")
                if href and "coid=" in href:
                    if href.startswith("http"):
                        full_url = href
                    elif href.startswith("/"):
                        full_url = "https://www.urban-research.tw" + href
                    else:
                        full_url = "https://www.urban-research.tw/" + href
                    seen_links.add(full_url)
            
            added = len(seen_links) - before
            print(f"    +{added} links, total={len(seen_links)}")
            
            if len(seen_links) >= MAX_OUTFITS:
                break

        links = list(seen_links)[:MAX_OUTFITS]
        print(f"Collected {len(links)} coordinate links to crawl\n")

        # Crawl detail pages
        for idx, url in enumerate(links, 1):
            print(f"[{idx}/{len(links)}] {url}")
            try:
                await page.goto(url, wait_until="networkidle", timeout=30000)
                await page.wait_for_timeout(int(PAGE_WAIT * 1000))
                
                items = await extract_outfit_items(page)
                print(f"  Extracted {len(items)} items:")
                for item in items[:3]:  # 只顯示前3項
                    print(f"    - {item[:100]}")
                
                parsed = parse_outfit(items)
                
                if not parsed:
                    print(f"  -> Could not parse outfit")
                    continue
                    
                parsed["id"] = url
                key = (
                    parsed["gender"],
                    parsed["top"]["type"],
                    parsed["top"]["color"],
                    parsed["bottom"]["type"],
                    parsed["bottom"]["color"],
                )
                
                if key in seen_data:
                    print(f"  -> Duplicate, skipping")
                    continue
                    
                seen_data.add(key)
                outfits.append(parsed)
                print(f"  ✓ #{len(outfits)} -> {parsed['top']['type']} / {parsed['bottom']['type']} ({parsed['top']['color']}, {parsed['bottom']['color']})")
                
                if len(outfits) >= MAX_OUTFITS:
                    break
                    
                await page.wait_for_timeout(int(PAGE_WAIT * 500))
                
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

        await browser.close()

    return outfits


def main():
    start = time.time()
    data = asyncio.run(crawl())
    elapsed = time.time() - start
    print(f"\nCollected {len(data)} outfits in {elapsed:.1f}s")
    OUTPUT_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
