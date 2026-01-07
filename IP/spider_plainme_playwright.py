import asyncio
import json
import re
import time
from pathlib import Path

from playwright.async_api import async_playwright

# =============================
# Config
# =============================

SEED_URL = "https://www.plain-me.com/men/mixandmatch/"
MAX_PAGES = 40  # seed pages to iterate (page=1..MAX_PAGES)
MAX_OUTFITS = 480  # total outfits to crawl
PAGE_WAIT = 1.0    # seconds to wait after load/scroll
SCROLL_ROUNDS = 3  # scroll times on seed page to load more cards
OUTPUT_FILE = Path("plainme_outfit_pairs.json")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) plainmeResearchBot/1.0",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
}

TOP_KEYWORDS = {
    "t-shirt": ["t恤", "tee", "上衣"],
    "shirt": ["襯衫", "衬衫", "shirt"],
    "hoodie": ["連帽", "帽t", "hoodie"],
    "sweater": ["毛衣", "針織", "針織衫", "針織衣"],
    "jacket": ["夾克", "外套", "風衣", "教練外套"],
    "coat": ["大衣", "長版外套"],
    "cardigan": ["開襟", "cardigan"],
    "vest": ["背心"],
}

BOTTOM_KEYWORDS = {
    "jeans": ["牛仔褲", "丹寧"],
    "pants": ["褲", "褲子", "長褲", "西裝褲", "寬褲", "神褲"],
    "shorts": ["短褲", "短裤"],
    "skirt": ["裙"],
}

COLORS = {
    "black": ["黑", "黑色"],
    "white": ["白", "白色"],
    "gray": ["灰", "灰色", "灰白"],
    "blue": ["藍", "蓝", "藍色", "蓝色", "水洗藍", "淺藍", "深藍"],
    "navy": ["深藍", "藏青"],
    "beige": ["米色", "卡其", "米白"],
    "brown": ["棕", "棕色", "咖啡"],
    "green": ["綠", "绿", "綠色", "绿色", "軍綠"],
    "red": ["紅", "红", "酒紅"],
    "pink": ["粉", "粉色", "粉紅"],
    "yellow": ["黃", "黄", "黃色", "黄色"],
    "orange": ["橙", "橙色", "橘", "橘色"],
    "purple": ["紫", "紫色"],
    "cream": ["奶油", "米白"],
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
    # Force gender to null because site does not cleanly separate male/female
    return None


async def scroll_page(page):
    for _ in range(SCROLL_ROUNDS):
        await page.mouse.wheel(0, 2000)
        await page.wait_for_timeout(int(PAGE_WAIT * 1000))


async def extract_outfit_items(page):
    """Extract texts per outfit item to keep colors local to each piece."""
    items = []

    # Try list items if present
    rows = await page.query_selector_all(".outfit-list li, .outfit-list-item")
    for r in rows:
        t = await r.inner_text()
        if t:
            items.append(t)

    # Fallback: pair names and specs by index
    if not items:
        names = await page.query_selector_all(".outfit-list-name")
        specs = await page.query_selector_all(".outfit-list-spec")
        limit = max(len(names), len(specs))
        for i in range(limit):
            parts = []
            if i < len(names):
                n = await names[i].inner_text()
                if n:
                    parts.append(n)
            if i < len(specs):
                s = await specs[i].inner_text()
                if s:
                    parts.append(s)
            if parts:
                items.append(" \n ".join(parts))

    # Last resort: whole body as one item
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
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(extra_http_headers=HEADERS)

        # collect detail links across pages
        print("Collecting outfit links...")
        for page_no in range(1, MAX_PAGES + 1):
            seed = f"{SEED_URL}&page={page_no}" if "?" in SEED_URL else f"{SEED_URL}?page={page_no}"
            await page.goto(seed, wait_until="networkidle")
            await page.wait_for_timeout(int(PAGE_WAIT * 1000))
            await scroll_page(page)
            html = await page.content()
            before = len(seen_links)
            for m in re.finditer(r'/men/mixandmatch-detail/[^"]+', html):
                seen_links.add("https://www.plain-me.com" + m.group(0))
            added = len(seen_links) - before
            print(f"  page {page_no}: +{added}, total={len(seen_links)}")
            if len(seen_links) >= MAX_OUTFITS:
                break

        links = list(seen_links)[:MAX_OUTFITS]
        print(f"Collected {len(links)} links")

        # crawl detail pages
        for idx, url in enumerate(links, 1):
            print(f"[{idx}/{len(links)}] {url}")
            await page.goto(url, wait_until="networkidle")
            await page.wait_for_timeout(int(PAGE_WAIT * 1000))
            items = await extract_outfit_items(page)
            parsed = parse_outfit(items)
            if not parsed:
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
                continue
            seen_data.add(key)
            outfits.append(parsed)
            if len(outfits) % 10 == 0:
                print(f"  ok #{len(outfits)} -> {parsed['top']['type']} / {parsed['bottom']['type']} ({parsed['top']['color']}, {parsed['bottom']['color']})")
            if len(outfits) >= MAX_OUTFITS:
                break
            await page.wait_for_timeout(int(PAGE_WAIT * 500))

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
