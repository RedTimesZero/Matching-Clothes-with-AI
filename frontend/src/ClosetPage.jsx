import { useEffect, useMemo, useState } from 'react'
import { supabase } from './supabaseClient.js'
import Shell from './Shell.jsx'

const BUCKET = 'images'

const CATEGORY_OPTIONS = [
  "capris", "jackets", "jeans", "leggings", "shirts", "shorts", "skirts", 
  "sweaters", "sweatshirts", "track pants", "trousers", "tshirts", "tunics"
]

const COLOR_OPTIONS = [
  "beige", "black", "blue", "brown", "burgundy", "charcoal", "coffee brown", 
  "cream", "gold", "green", "grey", "grey melange", "khaki", "lavender", 
  "lime green", "magenta", "maroon", "mauve", "multi", "mustard", "navy blue", 
  "nude", "off white", "olive", "orange", "peach", "pink", "purple", "red", 
  "rust", "sea green", "tan", "teal", "turquoise blue", "white", "yellow"
]

function normalizeCategory(raw) {
  const s = (raw || '').trim().toLowerCase()

  const map = {
    // 舊版本常見
    't-shirt': 'tshirts',
    'tshirts': 'tshirts',
    'shirt': 'shirts',
    'hoodie': 'sweatshirts',
    'sweater': 'sweaters',
    'jacket': 'jackets',
    'jeans': 'jeans',
    'wide pants': 'trousers',
    'pants': 'trousers',
    'shorts': 'shorts',
    'skirt': 'skirts',
    'dress': 'tunics',
    'other': 'other',
  }

  const v = map[s] || s
  return CATEGORY_OPTIONS.includes(v) ? v : 'other'
}

function normalizeColor(raw) {
  const s = (raw || '').trim().toLowerCase()

  const map = {
    // 你之前說的對應
    'dark blue': 'navy blue',
    'navy': 'navy blue',

    // 舊版本常見
    'gray': 'grey',
    'light blue': 'turquoise blue',
  }

  const v = map[s] || s
  return COLOR_OPTIONS.includes(v) ? v : 'multi'
}

function prettyLabel(s) {
  // 用於 UI 顯示：'navy blue' -> 'Navy Blue'
  return (s || '')
    .split(' ')
    .map(w => (w ? w[0].toUpperCase() + w.slice(1) : w))
    .join(' ')
}


// 把 DB row 轉成你卡片想用的格式
function rowToItem(row) {
  return {
    id: row.id,
    title: row.title,
    category: normalizeCategory(row.category),
    color: normalizeColor(row.color),
    worn: row.worn,
    image: row.image_url || '',
    image_path: row.image_path || null,
    created_at: row.created_at || null, // for sorting
  }
}


// 上傳衣櫃圖片到 Storage，回傳 publicUrl + path（path 會存 DB，之後刪檔用）
async function uploadClosetImage(file, userId) {
  const ext = file.name.split('.').pop()
  const path = `closet/${userId}/${crypto.randomUUID()}.${ext}`

  const { error: upErr } = await supabase.storage.from(BUCKET).upload(path, file, {
    upsert: false,
  })
  if (upErr) throw upErr

  const { data } = supabase.storage.from(BUCKET).getPublicUrl(path)
  return { publicUrl: data.publicUrl, path }
}
// main closetpage
export default function ClosetPage({ go, user }) {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  const [addingOpen, setAddingOpen] = useState(false)
  const [editingItem, setEditingItem] = useState(null)
  // 篩選state
  const [q, setQ] = useState('')
  const [cat, setCat] = useState('all')
  const [col, setCol] = useState('all')
  const [sort, setSort] = useState('newest') // newest | wornDesc | wornAsc | titleAsc
  //最少穿Top 3
  const recommended = [...items]
  .sort((a, b) => (a.worn ?? 0) - (b.worn ?? 0))
  .slice(0, 3)

  // visibleItems useMemo
  const visibleItems = useMemo(() => {
    const keyword = q.trim().toLowerCase()

    let arr = items.filter((it) => {
      const okQ =
        !keyword ||
        (it.title || '').toLowerCase().includes(keyword)

      const okCat =
        cat === 'all'
          ? true
          : cat === '__legacy'
            ? !CATEGORY_OPTIONS.includes(it.category)
            : it.category === cat

      const okCol =
        col === 'all'
          ? true
          : col === '__legacy'
            ? !COLOR_OPTIONS.includes(it.color)
            : it.color === col


      return okQ && okCat && okCol
    })

    // 排序
    arr = [...arr].sort((a, b) => {
      if (sort === 'wornDesc') return (b.worn || 0) - (a.worn || 0)
      if (sort === 'wornAsc') return (a.worn || 0) - (b.worn || 0)
      if (sort === 'titleAsc') return (a.title || '').localeCompare(b.title || '')
      // newest (預設)
      return new Date(b.created_at || 0) - new Date(a.created_at || 0)
    })

    return arr
  }, [items, q, cat, col, sort])

  // 進頁面先從 Supabase 把衣服抓回來
  useEffect(() => {
    if (!user) return

    let alive = true
    async function load() {
      setLoading(true)
      setError('')
      const { data, error } = await supabase
        .from('closet_items')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })

      if (!alive) return
      if (error) setError(error.message)
      setItems((data || []).map(rowToItem))
      setLoading(false)
    }

    load()
    return () => { alive = false }
  }, [user?.id])

  async function addCloth(form) {
    if (!user) return
    setBusy(true)
    setError('')

    try {
      let image_url = ''
      let image_path = null

      // 有選檔就上傳到 Storage
      if (form.file) {
        const uploaded = await uploadClosetImage(form.file, user.id)
        image_url = uploaded.publicUrl
        image_path = uploaded.path
      }

      // 寫入 DB
      const { data: row, error } = await supabase
        .from('closet_items')
        .insert({
          user_id: user.id,
          title: form.title || '未命名衣服',
          category: normalizeCategory(form.category),
          color: normalizeColor(form.color),
          worn: form.worn ?? 0,
          image_url,
          image_path,
        })
        .select()
        .single()

      if (error) throw error
      setItems(prev => [rowToItem(row), ...prev])
    } catch (e) {
      setError(e.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  async function updateCloth(id, form) {
    if (!user) return
    setBusy(true)
    setError('')

    try {
      const patch = {
        title: form.title || '未命名衣服',
        category: normalizeCategory(form.category),
        color: normalizeColor(form.color),
        worn: form.worn ?? 0,
      }

      // 若有換新圖片：再上傳一張，更新 image_url / image_path
      if (form.file) {
        const uploaded = await uploadClosetImage(form.file, user.id)
        patch.image_url = uploaded.publicUrl
        patch.image_path = uploaded.path
      }

      const { data: row, error } = await supabase
        .from('closet_items')
        .update(patch)
        .eq('id', id)
        .select()
        .single()

      if (error) throw error
      setItems(prev => prev.map(it => (it.id === id ? rowToItem(row) : it)))
    } catch (e) {
      setError(e.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  async function deleteCloth(id) {
    const ok = confirm("確定要刪除這件衣服嗎？")
    if (!ok) return

    setBusy(true)
    setError('')

    try {
      const target = items.find(x => x.id === id)

      const { error } = await supabase
        .from('closet_items')
        .delete()
        .eq('id', id)

      if (error) throw error

      // 可選：順便刪 Storage 的檔案（要你有建 delete policy）
      if (target?.image_path) {
        await supabase.storage.from(BUCKET).remove([target.image_path])
      }

      setItems(prev => prev.filter(it => it.id !== id))
    } catch (e) {
      setError(e.message || String(e))
    } finally {
      setBusy(false)
    }
  }
  async function listToMarket(it) {
    // 你 demo 的衣服欄位是 image，不一定叫 image_url，所以我做了 fallback
    const image_url = it.image_url ?? it.image ?? ''

    const { error } = await supabase.from('market_listings').insert({
      seller_id: user.id,
      title: it.title ?? '未命名商品',
      price: 300,              // 預設價
      size: 'M',               // 預設size is M
      condition: '9成新',       // 預設9成新
      tag: '衣櫃推薦',          // 讓交易區知道是推薦來的
      image_url,
      status: 'available',
    })

    if (error) {
      alert('上架失敗：' + error.message)
      return
    }

    alert('已上架到交易區！')
    go('market')
  }

  // 一鍵 +1（也同步 DB）
  async function wornPlusOne(item) {
    await updateCloth(item.id, { ...item, worn: (item.worn || 0) + 1 })
  }

  return (
  <Shell
    go={go}
    title="我的衣櫃"
    subtitle="上傳衣服照片、分類、顏色分析、穿著次數。"
  >
    {/* 上方主工具列：回首頁 + 新增 */}
    <div className="toolbar toolbarRow">
      <button className="btn btnGhost" onClick={() => go('home')}>
        ← 回主畫面
      </button>

      <div className="spacer" />

      <button className="btn btnPrimary" onClick={() => setAddingOpen(true)}>
        ＋ 新增
      </button>
    </div>

    {/* 篩選列 */}
    <div className="filterBar">
      <input
        className="control controlGrow"
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="搜尋衣服名稱（例如：skirt）"
      />

      <div className="filterRight">
        <select className="control" value={cat} onChange={(e) => setCat(e.target.value)}>
          <option value="all">All categories</option>
          {CATEGORY_OPTIONS.map(opt => (
            <option key={opt} value={opt}>{prettyLabel(opt)}</option>
          ))}

          <option value="__legacy">Legacy/Other</option>
        </select>

        <select className="control" value={col} onChange={(e) => setCol(e.target.value)}>
          <option value="all">All colors</option>
          {COLOR_OPTIONS.map(c => (
            <option key={c} value={c}>{prettyLabel(c)}</option>
          ))}

          <option value="__legacy">Legacy/Other</option>
        </select>

        <select className="control" value={sort} onChange={(e) => setSort(e.target.value)}>
          <option value="newest">Newest</option>
          <option value="wornDesc">Most worn</option>
          <option value="wornAsc">Least worn</option>
          <option value="titleAsc">Title A → Z</option>
        </select>

        <button
          className="btn btnGhost"
          onClick={() => { setQ(''); setCat('all'); setCol('all'); setSort('newest') }}
        >
          清除
        </button>

        <div className="filterCount">
          顯示 {visibleItems.length} / {items.length}
        </div>
      </div>
    </div>

    <div style={{ margin: '12px 0' }}>
      <h3 style={{ marginBottom: 8 }}>推薦你賣掉（最少穿）</h3>

      {recommended.map(it => (
        <div key={it.id} style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 8 }}>
          <div style={{ flex: 1 }}>
            {it.title}（穿過 {it.worn ?? 0} 次）
          </div>

          <button className="btn btnPrimary" onClick={() => listToMarket(it)}>
            上架到交易區
          </button>
        </div>
      ))}
    </div>

    {/* 原本的 error / loading / grid / modal 全部照舊 */}
    {error && <div style={{ margin: '10px 0', color: '#b00020' }}>{error}</div>}

    {loading ? (
      <div style={{ opacity: 0.7 }}>讀取中...</div>
    ) : (
      <div className="grid">
        <AddCard onClick={() => setAddingOpen(true)} />
        {visibleItems.map((it) => (
          <ClosetCard
            key={it.id}
            item={it}
            busy={busy}
            onEdit={() => setEditingItem(it)}
            onDelete={() => deleteCloth(it.id)}
            onWorn={() => wornPlusOne(it)}
          />
        ))}
      </div>
    )}

    {addingOpen && (
      <ClosetModal
        mode="add"
        onClose={() => setAddingOpen(false)}
        onSubmit={async (data) => {
          await addCloth(data)
          setAddingOpen(false)
        }}
      />
    )}

    {editingItem && (
      <ClosetModal
        mode="edit"
        initial={editingItem}
        onClose={() => setEditingItem(null)}
        onSubmit={async (data) => {
          await updateCloth(editingItem.id, data)
          setEditingItem(null)
        }}
      />
    )}
  </Shell>
)

}

function ClosetCard({ item, busy, onEdit, onDelete, onWorn }) {
  return (
    <div className="card">
      <img
        className="cardImg"
        alt={item.title}
        src={item.image || "https://images.unsplash.com/photo-1520975958225-8d56346d1b60?auto=format&fit=crop&w=1200&q=60"}
      />

      <div className="cardActions">
        <button className="iconBtn" onClick={onEdit} disabled={busy} title="編輯">Edit</button>
        <button className="iconBtn danger" onClick={onDelete} disabled={busy} title="刪除">Delete</button>
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

        <div className="toolbar" style={{ marginTop: 10 }}>
          <button className="btn btnGhost" onClick={onWorn} disabled={busy}>＋1 穿著</button>
        </div>
      </div>
    </div>
  )
}

function AddCard({ onClick }) {
  return (
    <button className="card addCard" onClick={onClick} aria-label="新增衣服">
      <div className="addCardInner">
        <div className="addPlus">＋</div>
        <div className="addTitle">新增衣服</div>
        <div className="addSub">上傳照片與基本資料</div>
      </div>
    </button>
  )
}

function ClosetModal({ mode, initial, onClose, onSubmit }) {
  const isEdit = mode === "edit"

  const [title, setTitle] = useState(initial?.title ?? '')
  const [category, setCategory] = useState(initial?.category ?? CATEGORY_OPTIONS[0])
  const [color, setColor] = useState(initial?.color ?? COLOR_OPTIONS[0])
  const [worn, setWorn] = useState(initial?.worn ?? 0)

  // 這裡的 preview 只拿來「畫面預覽」，真正會存的是上傳後的 publicUrl
  const [preview, setPreview] = useState(initial?.image ?? '')
  const [file, setFile] = useState(null)

  function handleFile(e) {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
  }

  // 避免 blob url 記憶體洩漏
  useEffect(() => {
    return () => {
      if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview)
    }
  }, [preview])

  return (
    <div className="modalBackdrop" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modalHead">
          <h3 className="modalTitle">{isEdit ? "編輯衣服" : "新增衣服到衣櫃"}</h3>
          <button className="btn btnGhost" onClick={onClose}>✕</button>
        </div>

        <div className="modalBody">
          <div className="formGrid">
            <div className="field fieldFull">
              <label>上傳照片</label>
              <input type="file" accept="image/*" onChange={handleFile} />
              {preview && <img className="previewImg" alt="preview" src={preview} />}
            </div>

            <div className="field">
              <label>衣服名稱</label>
              <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="例如：白色 T-shirt" />
            </div>

            <div className="field">
              <label>類別</label>
              <select value={category} onChange={(e) => setCategory(e.target.value)}>
                {CATEGORY_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
              </select>
            </div>

            <div className="field">
              <label>顏色</label>
              <select value={color} onChange={(e) => setColor(e.target.value)}>
                {COLOR_OPTIONS.map(c => (
                  <option key={c} value={c}>{c}</option>
                ))}
              </select>
            </div>

          </div>
        </div>

        <div className="modalFoot">
          <button className="btn btnGhost" onClick={onClose}>取消</button>
          <button
            className="btn btnPrimary"
            onClick={() => onSubmit({ 
              title, 
              category, 
              color: color || COLOR_OPTIONS[0], 
              worn, 
              file })
            }
          >
            {isEdit ? "儲存修改" : "新增到衣櫃"}
          </button>
        </div>
      </div>
    </div>
  )
}

