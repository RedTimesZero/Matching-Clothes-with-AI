import { useMemo, useState } from 'react'
import './App.css'

export default function App() {
  const [page, setPage] = useState('home')

  if (page === 'closet') return <ClosetPage go={setPage} />
  if (page === 'today') return <TodayPage go={setPage} />
  if (page === 'market') return <MarketPage go={setPage} />

  return (
    <div className="home">
      <div className="homeInner">
        <TopNav variant="dark" go={setPage} />

        <div className="heroContent">
          <div className="heroBox">
            <h1 className="heroTitle">Dress smarter.</h1>
            <p className="heroSubtitle">
              管理衣櫃、每日穿搭推薦、把很少穿的衣服快速整理成二手上架清單。
            </p>

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
   Shared Navbar
====================== */
function TopNav({ variant, go }) {
  const isLight = variant === 'light'
  return (
    <div className={`navbar ${isLight ? 'navbarLight' : ''}`} style={{ color: isLight ? '#4a2c1d' : '#fff' }}>
      <div className="brand" onClick={() => go('home')}>
        My Style Closet
      </div>
      <div className="navMenu">
        <button className="navBtn" onClick={() => go('closet')}>我的衣櫃</button>
        <button className="navBtn" onClick={() => go('today')}>今日穿搭推薦</button>
        <button className="navBtn" onClick={() => go('market')}>二手交易區</button>
      </div>
    </div>
  )
}

/* ======================
   Page Shell (統一風格)
====================== */
function Shell({ go, title, subtitle, children }) {
  return (
    <div className="shell">
      <TopNav variant="light" go={go} />
      <div className="container">
        <h1 className="pageTitle">{title}</h1>
        <p className="pageSubtitle">{subtitle}</p>
        {children}
      </div>
    </div>
  )
}

/* ======================
   Closet Page (先做統一風格示意)
====================== */
function ClosetPage({ go }) {
  const [items, setItems] = useState([
    { id: 'c1', title: '白色 T-shirt', badge: 'Top', color: '白色', worn: 5, image: '' },
    { id: 'c2', title: '牛仔褲', badge: 'Bottom', color: '藍色', worn: 2, image: '' },
    { id: 'c3', title: '深棕外套', badge: 'Outer', color: '棕色', worn: 1, image: '' },
  ])

  const [open, setOpen] = useState(false)

  function addCloth(newItem) {
    setItems(prev => [{ ...newItem, id: crypto.randomUUID() }, ...prev])
  }

  return (
    <Shell
      go={go}
      title="我的衣櫃"
      subtitle="之後會接：上傳衣服照片、分類、顏色分析、穿著次數。"
    >
      <div className="toolbar">
        <button className="btn btnGhost" onClick={() => go('home')}>← 回主畫面</button>
      </div>

      <div className="grid">
        {/* ✅ +號新增卡 */}
        <AddCard onClick={() => setOpen(true)} />

        {/* 原本衣服卡 */}
        {items.map((it) => (
          <ClosetCard key={it.id} item={it} />
        ))}
      </div>

      {open && (
        <AddClosetModal
          onClose={() => setOpen(false)}
          onSubmit={(data) => {
            addCloth(data)
            setOpen(false)
          }}
        />
      )}
    </Shell>
  )
}

function ClosetCard({ item }) {
  return (
    <div className="card">
      <img
        className="cardImg"
        alt={item.title}
        src={item.image || "https://images.unsplash.com/photo-1520975958225-8d56346d1b60?auto=format&fit=crop&w=1200&q=60"}
      />
      <div className="cardBody">
        <div className="cardTopRow">
          <p className="cardTitle">{item.title}</p>
          <span className="badge">{item.badge}</span>
        </div>
        <div className="meta">
          <span>{item.color}</span>
          <span>穿過 {item.worn} 次</span>
        </div>
      </div>
    </div>
  )
}

function AddCard({ onClick }) {
  return (
    <button
      className="card"
      onClick={onClick}
      style={{
        cursor: 'pointer',
        borderStyle: 'dashed',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        minHeight: 260,
        background: 'rgba(74, 44, 29, 0.02)'
      }}
      aria-label="新增衣服"
    >
      <div style={{ textAlign: 'center', padding: 18 }}>
        <div style={{ fontSize: 56, lineHeight: 1, color: 'rgba(74, 44, 29, 0.75)' }}>＋</div>
        <div style={{ marginTop: 8, fontWeight: 600 }}>新增衣服</div>
        <div style={{ marginTop: 6, fontSize: 13, opacity: 0.8 }}>
          上傳照片與基本資料
        </div>
      </div>
    </button>
  )
}

/* ✅ 上傳衣服 modal */
function AddClosetModal({ onClose, onSubmit }) {
  const [title, setTitle] = useState('')
  const [badge, setBadge] = useState('Top')
  const [color, setColor] = useState('白色')
  const [worn, setWorn] = useState(0)

  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')

  function handleFile(e) {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }

  return (
    <div className="modalBackdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modalHead">
          <h3 className="modalTitle">新增衣服到衣櫃</h3>
          <button className="btn btnGhost" onClick={onClose}>✕</button>
        </div>

        <div className="modalBody">
          <div className="formGrid">
            <div className="field" style={{ gridColumn: '1 / -1' }}>
              <label>上傳照片</label>
              <input type="file" accept="image/*" onChange={handleFile} />
              {preview && (
                <img
                  alt="preview"
                  src={preview}
                  style={{
                    marginTop: 10,
                    width: '100%',
                    height: 180,
                    objectFit: 'cover',
                    borderRadius: 12,
                    border: '1px solid rgba(74, 44, 29, 0.15)'
                  }}
                />
              )}
            </div>

            <div className="field">
              <label>衣服名稱</label>
              <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="例如：白色 T-shirt" />
            </div>

            <div className="field">
              <label>類別</label>
              <select value={badge} onChange={(e) => setBadge(e.target.value)}>
                <option value="Top">Top</option>
                <option value="Bottom">Bottom</option>
                <option value="Outer">Outer</option>
                <option value="Shoes">Shoes</option>
                <option value="Accessory">Accessory</option>
              </select>
            </div>

            <div className="field">
              <label>顏色</label>
              <input value={color} onChange={(e) => setColor(e.target.value)} placeholder="例如：白色 / 深棕" />
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
              onSubmit({
                title: title || '未命名衣服',
                badge,
                color,
                worn,
                image: preview, // 先用本機預覽 URL，之後可換成上傳到後端後的 URL
              })
            }}
          >
            新增到衣櫃
          </button>
        </div>
      </div>
    </div>
  )
}


/* ======================
   Today Page (統一風格示意)
====================== */
function TodayPage({ go }) {
  return (
    <Shell
      go={go}
      title="今日穿搭推薦"
      subtitle="Demo：先用假資料呈現推薦原因，之後可接模型/回饋按鈕。"
    >
      <div className="toolbar">
        <button className="btn btnGhost" onClick={() => go('home')}>← 回主畫面</button>
      </div>

      <div className="card">
        <img
          className="cardImg"
          alt="today"
          src="https://images.unsplash.com/photo-1521572163474-6864f9cf17ab?auto=format&fit=crop&w=1200&q=60"
        />
        <div className="cardBody">
          <div className="cardTopRow">
            <p className="cardTitle">推薦：白 T + 牛仔褲 + 深棕外套</p>
            <span className="badge">Today</span>
          </div>
          <div className="meta">
            <span>理由：中性色系好搭</span>
            <span>理由：外套很少穿</span>
            <span>理由：整體明暗平衡</span>
          </div>
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
   Market Page：商品卡 + 上架按鈕（可互動）
====================== */
function MarketPage({ go }) {
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

  const [items, setItems] = useState(initial)
  const [open, setOpen] = useState(false)

  function removeItem(id) {
    setItems((prev) => prev.filter((x) => x.id !== id))
  }

  function addItem(newItem) {
    setItems((prev) => [{ ...newItem, id: crypto.randomUUID() }, ...prev])
  }

  return (
    <Shell
      go={go}
      title="二手交易區"
      subtitle="Demo：用卡片呈現二手商品，支援「＋上架」新增商品與「下架」。"
    >
      <div className="toolbar">
        <button className="btn btnGhost" onClick={() => go('home')}>← 回主畫面</button>
        <button className="btn btnPrimary" onClick={() => setOpen(true)}>＋ 上架</button>
      </div>

      <div className="grid">
        {items.map((p) => (
          <div className="card" key={p.id}>
            <img className="cardImg" alt={p.title} src={p.image} />
            <div className="cardBody">
              <div className="cardTopRow">
                <p className="cardTitle">{p.title}</p>
                <span className="badge">{p.tag}</span>
              </div>

              <div className="meta">
                <span>尺寸：{p.size}</span>
                <span>狀態：{p.condition}</span>
              </div>

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
   Modal：上架表單
====================== */
function SellModal({ onClose, onSubmit }) {
  const [title, setTitle] = useState('')
  const [price, setPrice] = useState(300)
  const [size, setSize] = useState('M')
  const [condition, setCondition] = useState('9成新')
  const [image, setImage] = useState(
    'https://images.unsplash.com/photo-1520975947525-9a3f2e39e4e4?auto=format&fit=crop&w=1200&q=60'
  )
  const [tag, setTag] = useState('新上架')

  return (
    <div className="modalBackdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modalHead">
          <h3 className="modalTitle">上架二手商品</h3>
          <button className="btn btnGhost" onClick={onClose}>✕</button>
        </div>

        <div className="modalBody">
          <div className="formGrid">
            <div className="field">
              <label>商品名稱</label>
              <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="例如：黑色針織上衣" />
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

            <div className="field" style={{ gridColumn: '1 / -1' }}>
              <label>圖片網址（先用 URL demo，之後可改成上傳）</label>
              <input value={image} onChange={(e) => setImage(e.target.value)} />
            </div>

            <div className="field" style={{ gridColumn: '1 / -1' }}>
              <label>標籤</label>
              <input value={tag} onChange={(e) => setTag(e.target.value)} placeholder="例如：可議價/熱門/新上架" />
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
                image,
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
