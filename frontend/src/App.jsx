// React hooks：
// - useState：管理畫面狀態（目前在哪一頁、衣服清單、modal 開關、表單內容等）
// - useMemo：你目前有 import，但這段程式碼前半還沒用到；之後做「推薦計算/篩選」可用來加速
import { useEffect, useMemo, useState } from 'react'
import './App.css'

// 這是「衣服類別」的選項清單（全部用英文）
// 好處：
// 1) 前端統一類別字串，之後接模型/後端好對接
// 2) 下拉選單只需要 map 這個陣列即可
const CATEGORY_OPTIONS = [
  "t-shirt",
  "shirt",
  "hoodie",
  "sweater",
  "blouse",
  "jeans",
  "wide pants",
  "slim pants",
  "flare pants",
  "pants",
];

export default function App() {
  // page 控制目前顯示哪一個頁面（簡單版 router）
  // home / closet / today / market
  const [page, setPage] = useState('home')

  // 根據 page 切換不同頁面元件
  // go={setPage}：把「換頁功能」傳給子頁面使用
  if (page === 'closet') return <ClosetPage go={setPage} />
  if (page === 'today') return <TodayPage go={setPage} />
  if (page === 'market') return <MarketPage go={setPage} />

  // Home (Landing Page)
  // 這頁主要是展示滿版 Hero + 三個入口按鈕
  return (
    <div className="home">
      <div className="homeInner">
        {/* TopNav：共用導覽列（home 用深色版本） */}
        <TopNav variant="dark" go={setPage} />

        <div className="heroContent">
          <div className="heroBox">
            <h1 className="heroTitle">Dress smarter.</h1>
            <p className="heroSubtitle">
              管理衣櫃、每日穿搭推薦、把很少穿的衣服快速整理成二手上架清單。
            </p>

            {/* 三個主要功能入口 */}
            <div className="heroActions">
              <button className="heroCardBtn" onClick={() => setPage('closet')}>
                進入我的衣櫃
              </button>
              <button className="heroCardBtn" onClick={() => setPage('today')}>
                看今日推薦
              </button>
              <button className="heroCardBtn" onClick={() => setPage('market')}>
                前往二手交易
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

/* ======================
   Shared Navbar（共用導覽列）
   - variant: 'dark' or 'light' 用來決定顏色/樣式
   - go: setPage，點按鈕可切換頁面
====================== */
function TopNav({ variant, go }) {
  const isLight = variant === 'light'
  return (
    <div
      className={`navbar ${isLight ? 'navbarLight' : ''}`}
      style={{ color: isLight ? '#4a2c1d' : '#fff' }}
    >
      {/* 點品牌文字回首頁 */}
      <div className="brand" onClick={() => go('home')}>
        My Style Closet
      </div>

      {/* 三個導覽按鈕：切換頁面 */}
      <div className="navMenu">
        <button className="navBtn" onClick={() => go('closet')}>我的衣櫃</button>
        <button className="navBtn" onClick={() => go('today')}>今日穿搭推薦</button>
        <button className="navBtn" onClick={() => go('market')}>二手交易區</button>
      </div>
    </div>
  )
}

/* ======================
   Page Shell（統一版型）
   所有內頁（衣櫃/推薦/交易）都用同一個外框：
   - 上方 TopNav(light)
   - 內容 container
   - title / subtitle / children
====================== */
function Shell({ go, title, subtitle, children }) {
  return (
    <div className="shell">
      <TopNav variant="light" go={go} />
      <div className="container">
        <h1 className="pageTitle">{title}</h1>
        <p className="pageSubtitle">{subtitle}</p>
        {/* children = 每個頁面自己獨有的內容 */}
        {children}
      </div>
    </div>
  )
}

/* ======================
   Closet Page（我的衣櫃）
   目前是 demo 版：
   - items：衣服清單（存在前端 state）
   - addingOpen：新增 modal 是否開啟
   - editingItem：目前正在編輯的 item（或 null）
   - 支援：新增 / 編輯 / 刪除
====================== */
function ClosetPage({ go }) {
  // demo 初始衣服資料
  const [items, setItems] = useState([
    { id: 'c1', title: '白色 T-shirt', category: 't-shirt', color: 'white', worn: 5, image: '' },
    { id: 'c2', title: '牛仔褲', category: 'jeans', color: 'blue', worn: 2, image: '' },
    { id: 'c3', title: '深棕外套', category: 'sweater', color: 'brown', worn: 1, image: '' },
  ])

  // 新增 modal 的開關
  const [addingOpen, setAddingOpen] = useState(false)

  // 編輯中衣服（null = 沒有在編輯）
  const [editingItem, setEditingItem] = useState(null) // item or null

  // 新增衣服：把新衣服插到最前面（讓使用者一新增就看得到）
  function addCloth(newItem) {
    setItems(prev => [{ ...newItem, id: crypto.randomUUID() }, ...prev])
  }

  // 編輯衣服：用 id 找到那件衣服，覆蓋 patch 欄位
  function updateCloth(id, patch) {
    setItems(prev => prev.map(it => it.id === id ? { ...it, ...patch } : it))
  }

  // 刪除衣服：先 confirm 再刪（避免誤刪）
  function deleteCloth(id) {
    const ok = confirm("確定要刪除這件衣服嗎？")
    if (!ok) return
    setItems(prev => prev.filter(it => it.id !== id))
  }

  return (
    <Shell
      go={go}
      title="我的衣櫃"
      subtitle="上傳衣服照片、分類、顏色分析、穿著次數。"
    >
      {/* 工具列：回首頁 */}
      <div className="toolbar">
        <button className="btn btnGhost" onClick={() => go('home')}>← 回主畫面</button>
      </div>

      {/* 卡片網格：第一張是「＋新增」 */}
      <div className="grid">
        <AddCard onClick={() => setAddingOpen(true)} />

        {/* 衣服卡片列表 */}
        {items.map((it) => (
          <ClosetCard
            key={it.id}
            item={it}
            // 點 Edit：把這件衣服存到 editingItem，打開編輯 modal
            onEdit={() => setEditingItem(it)}
            // 點 Delete：刪除
            onDelete={() => deleteCloth(it.id)}
          />
        ))}
      </div>

      {/* 新增 Modal：addingOpen = true 才顯示 */}
      {addingOpen && (
        <ClosetModal
          mode="add"
          onClose={() => setAddingOpen(false)}
          onSubmit={(data) => {
            addCloth(data)
            setAddingOpen(false)
          }}
        />
      )}

      {/* 編輯 Modal：editingItem 有值才顯示 */}
      {editingItem && (
        <ClosetModal
          mode="edit"
          initial={editingItem}
          onClose={() => setEditingItem(null)}
          onSubmit={(data) => {
            updateCloth(editingItem.id, data)
            setEditingItem(null)
          }}
        />
      )}
    </Shell>
  )
}

/* ======================
   ClosetCard（單張衣服卡片）
   - 顯示圖片、標題、分類 badge、顏色、穿著次數
   - 右上角 Edit/Delete 讓使用者管理衣服
====================== */
function ClosetCard({ item, onEdit, onDelete }) {
  return (
    <div className="card">
      <img
        className="cardImg"
        alt={item.title}
        // 如果沒有 image（例如 demo 初始資料），就用一張預設圖
        src={item.image || "https://images.unsplash.com/photo-1520975958225-8d56346d1b60?auto=format&fit=crop&w=1200&q=60"}
      />

      {/* 卡片右上角：編輯 / 刪除 */}
      <div className="cardActions">
        <button className="iconBtn" onClick={onEdit} title="編輯">Edit</button>
        <button className="iconBtn danger" onClick={onDelete} title="刪除">Delete</button>
      </div>

      <div className="cardBody">
        <div className="cardTopRow">
          <p className="cardTitle">{item.title}</p>
          <span className="badge">{item.category}</span>
        </div>
        <div className="meta">
          <span>{item.color}</span>
          <span>穿過 {item.worn} 次</span>
        </div>
      </div>
    </div>
  )
}

/* ======================
   AddCard（＋新增卡片）
   - 長得像一張卡片，但點下去打開新增 modal
====================== */
function AddCard({ onClick }) {
  return (
    <button
      className="card addCard"
      onClick={onClick}
      aria-label="新增衣服"
    >
      <div className="addCardInner">
        <div className="addPlus">＋</div>
        <div className="addTitle">新增衣服</div>
        <div className="addSub">上傳照片與基本資料</div>
      </div>
    </button>
  )
}

/* ======================
   ClosetModal（新增 / 編輯共用表單）
   - mode: "add" or "edit"
   - initial: 編輯模式會帶入原本資料
   - onSubmit: 回傳表單資料給 ClosetPage 去更新 items
   注意：目前圖片只做「本機預覽 URL」，還沒上傳到後端
====================== */
function ClosetModal({ mode, initial, onClose, onSubmit }) {
  // 是否為編輯模式
  const isEdit = mode === "edit"

  // 表單欄位 state：新增模式預設空值；編輯模式用 initial 值初始化
  const [title, setTitle] = useState(initial?.title ?? '')
  const [category, setCategory] = useState(initial?.category ?? CATEGORY_OPTIONS[0])
  const [color, setColor] = useState(initial?.color ?? '')
  const [worn, setWorn] = useState(initial?.worn ?? 0)

  // preview：顯示上傳圖片的預覽（或編輯模式沿用舊圖）
  const [preview, setPreview] = useState(initial?.image ?? '') // 若不換圖就沿用

  // file：目前選到的檔案（現在只存著，之後接後端才會用到）
  const [file, setFile] = useState(null)

  // 選檔事件：把檔案轉成「可顯示的 URL」給 <img> 用
  function handleFile(e) {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }

  return (
    // 點背景就關閉（方便 demo）
    <div className="modalBackdrop" onClick={onClose}>
      {/* 點 modal 本體不要冒泡，避免點到背景觸發關閉 */}
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modalHead">
          <h3 className="modalTitle">{isEdit ? "編輯衣服" : "新增衣服到衣櫃"}</h3>
          <button className="btn btnGhost" onClick={onClose}>✕</button>
        </div>

        <div className="modalBody">
          <div className="formGrid">
            {/* 上傳圖片：佔滿整行 */}
            <div className="field fieldFull">
              <label>上傳照片</label>
              <input type="file" accept="image/*" onChange={handleFile} />
              {preview && (
                <img className="previewImg" alt="preview" src={preview} />
              )}
            </div>

            <div className="field">
              <label>衣服名稱</label>
              <input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="例如：白色 T-shirt"
              />
            </div>

            <div className="field">
              <label>類別</label>
              <select value={category} onChange={(e) => setCategory(e.target.value)}>
                {CATEGORY_OPTIONS.map(opt => (
                  <option key={opt} value={opt}>{opt}</option>
                ))}
              </select>
            </div>

            <div className="field">
              <label>顏色</label>
              <input
                value={color}
                onChange={(e) => setColor(e.target.value)}
                placeholder="例如：white / brown"
              />
            </div>

            <div className="field">
              <label>穿著次數</label>
              <input
                type="number"
                min="0"
                value={worn}
                onChange={(e) => setWorn(Number(e.target.value))}
              />
            </div>
          </div>
        </div>

        <div className="modalFoot">
          <button className="btn btnGhost" onClick={onClose}>取消</button>
          <button
            className="btn btnPrimary"
            onClick={() => {
              // 把表單資料回傳給父層（ClosetPage）處理新增/更新
              onSubmit({
                title: title || '未命名衣服',
                category,
                color: color || 'unknown',
                worn,
                // demo：用本機 preview URL（之後接後端再換成真正的圖片網址）
                image: preview,
              })
            }}
          >
            {isEdit ? "儲存修改" : "新增到衣櫃"}
          </button>
        </div>
      </div>
    </div>
  )
}

/* ======================
   Today Page（今日穿搭推薦）
   目前是 demo 頁：
   - 用固定假資料顯示「今日推薦」與「推薦理由」
   - 喜歡/不喜歡按鈕先做 UI，之後可以接：
     1) 模型回饋（like/dislike 記錄）
     2) 重新生成推薦
====================== */
function TodayPage({ go }) {
  return (
    <Shell
      go={go}
      title="今日穿搭推薦"
      subtitle="Demo：先用假資料呈現推薦原因，之後可接模型/回饋按鈕。"
    >
      {/* 工具列：回首頁 */}
      <div className="toolbar">
        <button className="btn btnGhost" onClick={() => go('home')}>← 回主畫面</button>
      </div>

      {/* Demo：一張推薦卡（圖片 + 推薦套裝 + 理由） */}
      <div className="card">
        <img
          className="cardImg"
          alt="today"
          // demo 用 unsplash 圖，之後可換成「你衣櫃的衣服組合圖」或「生成的 outfit 圖」
          src="https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&w=1200&q=60"
        />
        <div className="cardBody">
          <div className="cardTopRow">
            <p className="cardTitle">推薦：白 T + 牛仔褲 + 深棕外套</p>
            <span className="badge">Today</span>
          </div>

          {/* 推薦理由：先硬寫三條，之後可接模型輸出的 explainability */}
          <div className="meta">
            <span>理由：中性色系好搭</span>
            <span>理由：外套很少穿</span>
            <span>理由：整體明暗平衡</span>
          </div>

          {/* 回饋按鈕：現在是 UI，之後可以 onClick 送到後端 */}
          <div className="toolbar" style={{ marginTop: 12 }}>
            <button className="btn btnPrimary">👍 喜歡</button>
            <button className="btn btnGhost">👎 不喜歡</button>
          </div>
        </div>
      </div>
    </Shell>
  )
}

/* ======================
   Market Page（二手交易區）
   demo 特色：
   - 商品卡片列表（items）
   - 支援「＋上架」打開 SellModal
   - 支援「下架」把商品從列表移除
   實務上之後可接：
   - 後端資料庫（商品由 API 取得）
   - 買家聯絡資訊 / 私訊 / 下單
====================== */
function MarketPage({ go }) {
  // useMemo：把 initial 假資料固定住，不要每次 re-render 都重新產生一份
  //（不然如果未來你加更多狀態，會容易造成 items 被重置）
  const initial = useMemo(() => ([
    {
      id: 'a1',
      title: '黑色針織上衣',
      size: 'M',
      condition: '9成新',
      price: 380,
      image:
        'https://images.unsplash.com/photo-1520975682038-7d5b13e43a4a?auto=format&fit=crop&w=1200&q=60',
      tag: '熱門',
    },
    {
      id: 'a2',
      title: '米白襯衫',
      size: 'L',
      condition: '近全新',
      price: 520,
      image:
        'https://images.unsplash.com/photo-1520975869018-5d3b2f5a3c30?auto=format&fit=crop&w=1200&q=60',
      tag: '推薦',
    },
    {
      id: 'a3',
      title: '牛仔外套',
      size: 'M',
      condition: '8成新',
      price: 650,
      image:
        'https://images.unsplash.com/photo-1512436991641-6745cdb1723f?auto=format&fit=crop&w=1200&q=60',
      tag: '可議價',
    },
  ]), [])

  // items：目前畫面上的商品清單（demo 存在前端 state）
  const [items, setItems] = useState(initial)

  // open：控制 SellModal 是否顯示
  const [open, setOpen] = useState(false)

  // 下架功能：用 filter 移除 id 對應商品
  function removeItem(id) {
    setItems((prev) => prev.filter((x) => x.id !== id))
  }

  // 上架功能：把新商品插到最前面（新增後立刻看得到）
  function addItem(newItem) {
    setItems((prev) => [{ ...newItem, id: crypto.randomUUID() }, ...prev])
  }

  return (
    <Shell
      go={go}
      title="二手交易區"
      subtitle="Demo：用卡片呈現二手商品，支援「＋上架」新增商品與「下架」。"
    >
      {/* 工具列：回首頁 + 上架按鈕 */}
      <div className="toolbar">
        <button className="btn btnGhost" onClick={() => go('home')}>← 回主畫面</button>
        <button className="btn btnPrimary" onClick={() => setOpen(true)}>＋ 上架</button>
      </div>

      {/* 商品列表：grid 版面卡片 */}
      <div className="grid">
        {items.map((p) => (
          <div className="card" key={p.id}>
            <img className="cardImg" alt={p.title} src={p.image} />
            <div className="cardBody">
              <div className="cardTopRow">
                <p className="cardTitle">{p.title}</p>
                <span className="badge">{p.tag}</span>
              </div>

              {/* 商品資訊 */}
              <div className="meta">
                <span>尺寸：{p.size}</span>
                <span>狀態：{p.condition}</span>
              </div>

              {/* 價格 + 下架 */}
              <div className="priceRow">
                <span className="price">NT$ {p.price}</span>
                <button className="btn btnGhost" onClick={() => removeItem(p.id)}>
                  下架
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* 上架 Modal：open=true 才顯示 */}
      {open && (
        <SellModal
          onClose={() => setOpen(false)}
          onSubmit={(data) => {
            addItem(data)
            setOpen(false)
          }}
        />
      )}
    </Shell>
  )
}

/* ======================
   SellModal（上架表單：支援上傳圖片 + 預覽）
   - 主要用「上傳圖片」：使用者體驗更像真的二手平台
   - 保留「圖片網址」：如果你想快速 demo 或沒準備照片
====================== */
function SellModal({ onClose, onSubmit }) {
  const [title, setTitle] = useState('')
  const [price, setPrice] = useState(300)
  const [size, setSize] = useState('M')
  const [condition, setCondition] = useState('9成新')
  const [tag, setTag] = useState('新上架')

  // ✅ 圖片：支援「上傳」與「URL」
  const [imageUrl, setImageUrl] = useState(
    'https://images.unsplash.com/photo-1520975947525-9a3f2e39e4e4?auto=format&fit=crop&w=1200&q=60'
  )
  const [preview, setPreview] = useState('') // 優先顯示上傳的預覽
  const [file, setFile] = useState(null)

  function handleFile(e) {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)

    // 產生本機預覽 URL（demo 用）
    const url = URL.createObjectURL(f)
    setPreview(url)
  }

  // ✅ 避免記憶體洩漏：換圖或關閉 modal 時，把舊的 objectURL revoke 掉
  useEffect(() => {
    return () => {
      if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview)
    }
  }, [preview])

  // 最終送出的 image：優先用上傳預覽（如果有），否則用 URL
  const finalImage = preview || imageUrl

  return (
    <div className="modalBackdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modalHead">
          <h3 className="modalTitle">上架二手商品</h3>
          <button className="btn btnGhost" onClick={onClose}>✕</button>
        </div>

        <div className="modalBody">
          <div className="formGrid">
            {/* ✅ 上傳圖片（跨整行） */}
            <div className="field fieldFull">
              <label>上傳商品照片</label>
              <input type="file" accept="image/*" onChange={handleFile} />
              {(preview || imageUrl) && (
                <img className="previewImg" alt="preview" src={finalImage} />
              )}
            </div>

            <div className="field">
              <label>商品名稱</label>
              <input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="例如：黑色針織上衣"
              />
            </div>

            <div className="field">
              <label>價格（NT$）</label>
              <input
                type="number"
                value={price}
                onChange={(e) => setPrice(Number(e.target.value))}
                min="0"
              />
            </div>

            <div className="field">
              <label>尺寸</label>
              <select value={size} onChange={(e) => setSize(e.target.value)}>
                <option>S</option>
                <option>M</option>
                <option>L</option>
                <option>XL</option>
              </select>
            </div>

            <div className="field">
              <label>狀態</label>
              <select value={condition} onChange={(e) => setCondition(e.target.value)}>
                <option>近全新</option>
                <option>9成新</option>
                <option>8成新</option>
                <option>有使用痕跡</option>
              </select>
            </div>

            {/* ✅ 保留 URL 欄位：當你沒上傳圖片時，就會用這個 */}
            <div className="field fieldFull">
              <label>圖片網址（備用）</label>
              <input
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
                placeholder="貼圖片網址（可選）"
              />
              <div style={{ fontSize: 12, opacity: 0.75, marginTop: 6 }}>
                有上傳圖片時會優先使用「上傳的照片」；沒上傳才會用這個網址。
              </div>
            </div>

            <div className="field fieldFull">
              <label>標籤</label>
              <input
                value={tag}
                onChange={(e) => setTag(e.target.value)}
                placeholder="例如：可議價/熱門/新上架"
              />
            </div>
          </div>
        </div>

        <div className="modalFoot">
          <button className="btn btnGhost" onClick={onClose}>取消</button>
          <button
            className="btn btnPrimary"
            onClick={() =>
              onSubmit({
                title: title || '未命名商品',
                price,
                size,
                condition,
                image: finalImage, // ✅ 這裡就會存到卡片用的 image
                tag,
              })
            }
          >
            確認上架
          </button>
        </div>
      </div>
    </div>
  )
}
