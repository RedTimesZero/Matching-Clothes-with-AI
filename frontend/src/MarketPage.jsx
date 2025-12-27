import { useEffect, useMemo, useState } from 'react'
import { supabase } from './supabaseClient.js'
import AddListing from './AddListing.jsx'

/* ======================
   Market Page（二手交易區）
   demo 特色：
   - 商品卡片列表（items）
   - 支援「＋上架」打開 SellModal
   - 支援「Edit/Delete」把商品從列表移除
   - + 留言區
   實務上之後可接：
   - 後端資料庫（商品由 API 取得）
   - 買家聯絡資訊 / 私訊 / 下單
====================== */
export default function MarketPage({ go, user }) {
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
      seller: 'Alice',            // ✅ 別人上架
      comments: [
        { id: 'cm1', author: 'Penny', text: '請問有實穿照嗎？', time: Date.now() - 1000 * 60 * 30 },
      ],
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
      seller: 'You',              // ✅ 你上架（可 edit/delete）
      comments: [],
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
      seller: 'Bob',              // ✅ 別人上架
      comments: [],
    },
  ]), [])

  const [items, setItems] = useState(initial)

  // 新增/編輯 Modal 控制
  const [modalOpen, setModalOpen] = useState(false)
  const [editingItem, setEditingItem] = useState(null) // item or null

  // 商品詳情（留言區）
  const [selectedItem, setSelectedItem] = useState(null) // item or null

  function addItem(newItem) {
    setItems((prev) => [{ ...newItem, id: crypto.randomUUID(), seller: 'You', comments: [] }, ...prev])
  }

  function updateItem(id, patch) {
    setItems((prev) => prev.map((x) => (x.id === id ? { ...x, ...patch } : x)))
  }

  function deleteItem(id) {
    const ok = confirm('確定要刪除（下架）這個商品嗎？')
    if (!ok) return
    setItems((prev) => prev.filter((x) => x.id !== id))
  }

  function addComment(productId, text) {
    setItems((prev) =>
      prev.map((p) => {
        if (p.id !== productId) return p
        const next = {
          ...p,
          comments: [
            ...(p.comments || []),
            { id: crypto.randomUUID(), author: 'You', text, time: Date.now() },
          ],
        }
        return next
      })
    )

    // 讓「詳情 modal」也同步顯示最新留言（因為 selectedItem 是舊物件）
    setSelectedItem((cur) => {
      if (!cur || cur.id !== productId) return cur
      return {
        ...cur,
        comments: [
          ...(cur.comments || []),
          { id: crypto.randomUUID(), author: 'You', text, time: Date.now() },
        ],
      }
    })
  }

  return (
    <Shell
      go={go}
      title="二手交易區"
      subtitle="Demo：支援上架、自己商品可編輯/刪除，別人商品可進入詳情留言。"
    >
      <div className="toolbar">
        <button className="btn btnGhost" onClick={() => go('home')}>← 回主畫面</button>
        <button
          className="btn btnPrimary"
          onClick={() => {
            setEditingItem(null)
            setModalOpen(true)
          }}
        >
          ＋ 上架
        </button>
      </div>

      <div className="grid">
        {items.map((p) => (
          <ProductCard
            key={p.id}
            item={p}
            isMine={p.seller === 'You'}
            onOpen={() => setSelectedItem(p)}
            onEdit={() => {
              setEditingItem(p)
              setModalOpen(true)
            }}
            onDelete={() => deleteItem(p.id)}
          />
        ))}
      </div>

      {/* 新增/編輯共用 Modal */}
      {modalOpen && (
        <ProductModal
          mode={editingItem ? 'edit' : 'add'}
          initial={editingItem}
          onClose={() => setModalOpen(false)}
          onSubmit={(data) => {
            if (editingItem) {
              updateItem(editingItem.id, data)
            } else {
              addItem(data)
            }
            setModalOpen(false)
            setEditingItem(null)
          }}
        />
      )}

      {/* 商品詳情（留言區） */}
      {selectedItem && (
        <ProductDetailModal
          item={selectedItem}
          onClose={() => setSelectedItem(null)}
          onAddComment={(text) => addComment(selectedItem.id, text)}
        />
      )}
    </Shell>
  )
}

/* ======================
   ProductCard：交易區卡片
   - 自己的商品：顯示 Edit/Delete
   - 別人的商品：顯示「查看/留言」
====================== */
function ProductCard({ item, isMine, onOpen, onEdit, onDelete }) {
  return (
    <div className="card">
      <img className="cardImg" alt={item.title} src={item.image} />

      {/* 右上角動作區 */}
      <div className="cardActions">
        {isMine ? (
          <>
            <button className="iconBtn" onClick={onEdit} title="編輯">Edit</button>
            <button className="iconBtn danger" onClick={onDelete} title="刪除">Delete</button>
          </>
        ) : (
          <button className="iconBtn" onClick={onOpen} title="查看與留言">View</button>
        )}
      </div>

      <div className="cardBody">
        <div className="cardTopRow">
          <p className="cardTitle">{item.title}</p>
          <span className="badge">{item.tag}</span>
        </div>

        <div className="meta">
          <span>賣家：{item.seller}</span>
          <span>尺寸：{item.size}</span>
          <span>狀態：{item.condition}</span>
          <span>留言：{(item.comments || []).length}</span>
        </div>

        <div className="priceRow">
          <span className="price">NT$ {item.price}</span>
          <button className="btn btnGhost" onClick={onOpen}>
            {isMine ? '查看' : '查看 / 留言'}
          </button>
        </div>
      </div>
    </div>
  )
}

/* ======================
   ProductModal：上架/編輯共用表單（含上傳圖片預覽）
====================== */
function ProductModal({ mode, initial, onClose, onSubmit }) {
  const isEdit = mode === 'edit'

  const [title, setTitle] = useState(initial?.title ?? '')
  const [price, setPrice] = useState(initial?.price ?? 300)
  const [size, setSize] = useState(initial?.size ?? 'M')
  const [condition, setCondition] = useState(initial?.condition ?? '9成新')
  const [tag, setTag] = useState(initial?.tag ?? '新上架')

  // 圖片：支援上傳與 URL（備用）
  const [imageUrl, setImageUrl] = useState(initial?.image ?? '')
  const [preview, setPreview] = useState(initial?.image ?? '')
  const [file, setFile] = useState(null)

  function handleFile(e) {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    const url = URL.createObjectURL(f)
    setPreview(url)
  }

  useEffect(() => {
    return () => {
      if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview)
    }
  }, [preview])

  const finalImage = preview || imageUrl

  return (
    <div className="modalBackdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modalHead">
          <h3 className="modalTitle">{isEdit ? '編輯商品' : '上架二手商品'}</h3>
          <button className="btn btnGhost" onClick={onClose}>✕</button>
        </div>

        <div className="modalBody">
          <div className="formGrid">
            <div className="field fieldFull">
              <label>上傳商品照片</label>
              <input type="file" accept="image/*" onChange={handleFile} />
              {finalImage && <img className="previewImg" alt="preview" src={finalImage} />}
            </div>

            <div className="field">
              <label>商品名稱</label>
              <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="例如：黑色針織上衣" />
            </div>

            <div className="field">
              <label>價格（NT$）</label>
              <input type="number" value={price} onChange={(e) => setPrice(Number(e.target.value))} min="0" />
            </div>

            <div className="field">
              <label>尺寸</label>
              <select value={size} onChange={(e) => setSize(e.target.value)}>
                <option>S</option><option>M</option><option>L</option><option>XL</option>
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

            <div className="field fieldFull">
              <label>圖片網址（備用）</label>
              <input value={imageUrl} onChange={(e) => setImageUrl(e.target.value)} placeholder="可留空" />
            </div>

            <div className="field fieldFull">
              <label>標籤</label>
              <input value={tag} onChange={(e) => setTag(e.target.value)} placeholder="例如：可議價/熱門/新上架" />
            </div>
          </div>
        </div>

        <div className="modalFoot">
          <button className="btn btnGhost" onClick={onClose}>取消</button>
          <button
            className="btn btnPrimary"
            onClick={() => onSubmit({
              title: title || '未命名商品',
              price,
              size,
              condition,
              image: finalImage,
              tag,
            })}
          >
            {isEdit ? '儲存修改' : '確認上架'}
          </button>
        </div>
      </div>
    </div>
  )
}

/* ======================
   ProductDetailModal：商品詳情 + 留言區
   - 點進去看商品資訊
   - 留言列表 + 新增留言
   注意：目前留言只存在前端 state（重整會消失）
====================== */
function ProductDetailModal({ item, onClose, onAddComment }) {
  const [text, setText] = useState('')

  function submit() {
    const t = text.trim()
    if (!t) return
    onAddComment(t)
    setText('')
  }

  return (
    <div className="modalBackdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modalHead">
          <h3 className="modalTitle">商品詳情</h3>
          <button className="btn btnGhost" onClick={onClose}>✕</button>
        </div>

        <div className="modalBody">
          <img className="previewImg" alt={item.title} src={item.image} />

          <div style={{ marginTop: 12 }}>
            <div className="cardTopRow">
              <p className="cardTitle" style={{ margin: 0 }}>{item.title}</p>
              <span className="badge">{item.tag}</span>
            </div>

            <div className="meta" style={{ marginTop: 8 }}>
              <span>賣家：{item.seller}</span>
              <span>尺寸：{item.size}</span>
              <span>狀態：{item.condition}</span>
              <span className="price">NT$ {item.price}</span>
            </div>
          </div>

          <hr style={{ margin: '16px 0', opacity: 0.2 }} />

          {/* 留言區 */}
          <div>
            <h4 style={{ margin: '0 0 10px 0' }}>留言區</h4>

            <div style={{ display: 'grid', gap: 10 }}>
              {(item.comments || []).length === 0 ? (
                <div style={{ opacity: 0.7, fontSize: 14 }}>目前還沒有留言。</div>
              ) : (
                (item.comments || []).map((c) => (
                  <div
                    key={c.id}
                    style={{
                      border: '1px solid rgba(74, 44, 29, 0.15)',
                      borderRadius: 12,
                      padding: 10,
                      background: 'rgba(74,44,29,0.02)'
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                      <strong style={{ fontSize: 14 }}>{c.author}</strong>
                      <span style={{ fontSize: 12, opacity: 0.65 }}>
                        {new Date(c.time).toLocaleString()}
                      </span>
                    </div>
                    <div style={{ marginTop: 6, fontSize: 14 }}>{c.text}</div>
                  </div>
                ))
              )}
            </div>

            <div className="toolbar" style={{ marginTop: 12 }}>
              <input
                style={{ flex: 1 }}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder="輸入留言（例如：請問可小議嗎？）"
              />
              <button className="btn btnPrimary" onClick={submit}>送出</button>
            </div>
          </div>
        </div>

        <div className="modalFoot">
          <button className="btn btnGhost" onClick={onClose}>關閉</button>
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


