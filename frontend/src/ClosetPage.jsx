import { useEffect, useState } from 'react'
import { supabase } from './supabaseClient.js'
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
/* ======================
   Closet Page（我的衣櫃）
   目前是 demo 版：
   - items：衣服清單（存在前端 state）
   - addingOpen：新增 modal 是否開啟
   - editingItem：目前正在編輯的 item（或 null）
   - 支援：新增 / 編輯 / 刪除
====================== */
export default function ClosetPage({ go, user }) {
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

