import json

# ----------------------------
# 顏色關鍵字（保守版）
# ----------------------------
COLORS = {
    'black', 'white', 'red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink',
    'brown', 'gray', 'grey', 'navy', 'cream', 'beige', 'gold', 'silver', 'rose',
    'burgundy', 'maroon', 'olive', 'khaki', 'tan', 'ivory'
}

# ----------------------------
# 從 item metadata 的 categories 判 gender（唯一正確來源）
# ----------------------------
def infer_gender_from_categories(item_meta):
    categories = item_meta.get("categories", [])
    if "Women's Fashion" in categories:
        return "female"
    if "Men's Fashion" in categories:
        return "male"
    return None


# ----------------------------
# 顏色抽取（沿用你的策略）
# ----------------------------
def extract_color(description, title):
    text = ((description or "") + " " + (title or "")).lower()
    for color in COLORS:
        if color in text:
            return color
    return "unknown"


# ----------------------------
# Top / Bottom 類型關鍵字
# ----------------------------
TOP_KEYWORDS = {
    "t-shirt": ["t-shirt", "tee", "tshirt"],
    "shirt": ["shirt"],
    "blouse": ["blouse"],
    "hoodie": ["hoodie", "sweatshirt"],
    "sweater": ["sweater", "knit", "jumper", "pullover"],
    "cardigan": ["cardigan"],
    "jacket": ["jacket", "blazer"],
    "coat": ["coat", "trench", "overcoat"],
    "vest": ["vest"]
}

BOTTOM_KEYWORDS = {
    "jeans": ["jeans", "denim"],
    "pants": ["pants", "trousers", "slacks", "chinos"],
    "shorts": ["shorts"],
    "skirt": ["skirt"],
    "leggings": ["legging"],
}


def get_item_type(text, is_top=True):
    text = text.lower()
    keywords = TOP_KEYWORDS if is_top else BOTTOM_KEYWORDS
    for item_type, kws in keywords.items():
        for kw in kws:
            if kw in text:
                return item_type
    return "unknown"


# ----------------------------
# 主程式
# ----------------------------
def main():
    print("Loading data...")

    with open("IP/polyvore/train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    with open("IP/polyvore/item_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"Loaded {len(train_data)} outfits")
    print(f"Loaded {len(metadata)} items")

    outfit_pairs = []

    for idx, outfit in enumerate(train_data):
        if idx % 1000 == 0:
            print(f"Processing {idx}/{len(train_data)}")

        items = outfit.get("items", [])

        tops = []
        bottoms = []

        for it in items:
            item_id = it.get("item_id")
            if item_id not in metadata:
                continue

            item_meta = metadata[item_id]
            cat = item_meta.get("semantic_category", "")

            if cat in ["tops", "outerwear"]:
                tops.append(item_meta)
            elif cat == "bottoms":
                bottoms.append(item_meta)
            elif cat == "dresses":
                # dress 視為 top-only outfit
                tops.append(item_meta)

        if not tops:
            continue

        top = tops[0]
        bottom = bottoms[0] if bottoms else None

        # -------- type --------
        top_text = (top.get("title", "") + " " + top.get("description", ""))
        top_type = get_item_type(top_text, is_top=True)

        if bottom:
            bottom_text = (bottom.get("title", "") + " " + bottom.get("description", ""))
            bottom_type = get_item_type(bottom_text, is_top=False)
        else:
            bottom_type = None

        # -------- color --------
        top_color = extract_color(top.get("description", ""), top.get("title", ""))
        bottom_color = extract_color(
            bottom.get("description", ""), bottom.get("title", "")
        ) if bottom else None

        # -------- gender（性別必須一致或其中一個為空）--------
        top_gender = infer_gender_from_categories(top)
        bottom_gender = infer_gender_from_categories(bottom) if bottom else None

        # 性別邏輯：兩個都有時必須一致；一個有一個沒有時用有的那個；都沒有就跳過
        if top_gender and bottom_gender:
            # 兩個都有，必須一致
            if top_gender != bottom_gender:
                continue
            gender = top_gender
        elif top_gender:
            # 只有 top 有
            gender = top_gender
        elif bottom_gender:
            # 只有 bottom 有
            gender = bottom_gender
        else:
            # 都沒有
            continue

        # -------- 跳過 null 或 unknown 數據 --------
        if not bottom or not top:
            continue

        if gender == "unknown" or top_type == "unknown" or top_color == "unknown":
            continue

        if bottom_type == "unknown" or bottom_color == "unknown":
            continue

        # -------- 最終輸出 --------
        pair = {
            "gender": gender,
            "top": {
                "type": top_type,
                "color": top_color
            }
        }

        if bottom:
            pair["bottom"] = {
                "type": bottom_type,
                "color": bottom_color
            }
        else:
            pair["bottom"] = None

        outfit_pairs.append(pair)

    print(f"\nGenerated {len(outfit_pairs)} outfit pairs")

    output_path = "IP/polyvore_outfit_pairs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outfit_pairs, f, indent=2, ensure_ascii=False)

    print(f"Saved to {output_path}")
    print("\nSample outputs:")
    for p in outfit_pairs[:5]:
        print(json.dumps(p, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
