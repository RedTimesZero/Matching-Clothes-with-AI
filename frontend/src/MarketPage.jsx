import { useEffect, useMemo, useState } from 'react'
import { supabase } from './supabaseClient.js'
import Shell from './Shell.jsx'


const BUCKET = 'images' // Supabase Storage bucket 名稱

async function uploadMarketImage(file, userId) {
  // 需要：Supabase Storage 建 bucket：images，先設 public
  const ext = file.name.split('.').pop()
  const path = `market/${userId}/${crypto.randomUUID()}.${ext}`

  const { error: upErr } = await supabase.storage.from(BUCKET).upload(path, file, { upsert: false })
  if (upErr) throw upErr

  const { data } = supabase.storage.from(BUCKET).getPublicUrl(path)
  return data.publicUrl
}

export default function MarketPage({ go, user }) {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')

  // 新增/編輯 modal
  const [modalOpen, setModalOpen] = useState(false)
  const [editingId, setEditingId] = useState(null)

  // 詳情/留言（用 id，避免 selectedItem 變舊）
  const [selectedId, setSelectedId] = useState(null)
  const selectedItem = useMemo(
    () => items.find((x) => x.id === selectedId) || null,
    [items, selectedId]
  )

  // 留言狀態
  const [comments, setComments] = useState([])
  const [commentsLoading, setCommentsLoading] = useState(false)
  const [commentBusy, setCommentBusy] = useState(false)

  // 搜尋
  const [q, setQ] = useState('')
  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase()
    if (!s) return items
    return items.filter((x) =>
      (x.title || '').toLowerCase().includes(s) ||
      (x.tag || '').toLowerCase().includes(s) ||
      (x.size || '').toLowerCase().includes(s)
    )
  }, [items, q])

  // ====== Listings CRUD ======
  async function fetchListings() {
    setLoading(true)
    setError('')
    const { data, error } = await supabase
      .from('market_listings')
      .select('id,title,price,size,condition,tag,image_url,seller_id,status,created_at')
      .order('created_at', { ascending: false })

    if (error) {
      setError(error.message)
      setItems([])
      setLoading(false)
      return
    }

    // 映射成你前端想用的格式（image/seller）
    const mapped = (data || []).map((r) => ({
      id: r.id,
      title: r.title,
      price: r.price,
      size: r.size,
      condition: r.condition,
      tag: r.tag,
      image: r.image_url,
      seller_id: r.seller_id,
      status: r.status || 'available', 
      seller: r.seller_id === user?.id ? 'You' : (r.seller_id ? r.seller_id.slice(0, 6) : 'Unknown'),
      created_at: r.created_at,
    }))

    setItems(mapped)
    setLoading(false)
  }
async function setStatus(id, status) {
  if (!user) return alert('請先登入')
  setBusy(true)
  setError('')

  try {
    const { error } = await supabase
      .from('market_listings')
      .update({ status })
      .eq('id', id)
      .eq('seller_id', user.id) // ✅ 只能改自己的

    if (error) throw error

    await fetchListings()
  } catch (e) {
    setError(e.message || String(e))
  } finally {
    setBusy(false)
  }
}

  async function createListing(form) {
    if (!user) return alert('請先登入才能上架')
    setBusy(true)
    setError('')

    try {
      // 圖片：有 file 就上傳到 Storage，否則用 imageUrl（網址）
      let image_url = form.imageUrl || ''
      if (form.file) image_url = await uploadMarketImage(form.file, user.id)

      const { error } = await supabase.from('market_listings').insert({
        seller_id: user.id,
        title: form.title || '未命名商品',
        price: Number(form.price) || 0,
        size: form.size || 'M',
        condition: form.condition || '9成新',
        tag: form.tag || '新上架',
        image_url,
      })
      if (error) throw error

      await fetchListings()
    } catch (e) {
      setError(e.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  async function updateListing(id, form) {
    if (!user) return alert('請先登入才能編輯')
    setBusy(true)
    setError('')

    try {
      let image_url = form.imageUrl || ''
      if (form.file) image_url = await uploadMarketImage(form.file, user.id)

      const { error } = await supabase
        .from('market_listings')
        .update({
          title: form.title || '未命名商品',
          price: Number(form.price) || 0,
          size: form.size || 'M',
          condition: form.condition || '9成新',
          tag: form.tag || '新上架',
          image_url,
        })
        .eq('id', id)
        .eq('seller_id', user.id) // 防止改到別人的

      if (error) throw error

      await fetchListings()
    } catch (e) {
      setError(e.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  async function deleteListing(id) {
    if (!user) return alert('請先登入')
    const ok = confirm('確定要刪除（下架）這個商品嗎？')
    if (!ok) return

    setBusy(true)
    setError('')
    try {
      const { error } = await supabase
        .from('market_listings')
        .delete()
        .eq('id', id)
        .eq('seller_id', user.id)

      if (error) throw error

      // 若正在看這個商品詳情，順便關掉
      if (selectedId === id) setSelectedId(null)
      await fetchListings()
    } catch (e) {
      setError(e.message || String(e))
    } finally {
      setBusy(false)
    }
  }

  // ====== Comments ======
  async function fetchComments(listingId) {
    if (!listingId) return
    setCommentsLoading(true)
    const { data, error } = await supabase
      .from('comments')
      .select('id,text,created_at,author_id')
      .eq('listing_id', listingId)
      .order('created_at', { ascending: true })

    if (error) {
      setComments([])
      setCommentsLoading(false)
      return
    }

    setComments((data || []).map((c) => ({
      id: c.id,
      text: c.text,
      time: c.created_at,
      author_id: c.author_id,
      author: c.author_id === user?.id ? 'You' : (c.author_id ? c.author_id.slice(0, 6) : 'Unknown'),
    })))
    setCommentsLoading(false)
  }

  async function addComment(listingId, text) {
    if (!user) return alert('請先登入才能留言')
    const t = text.trim()
    if (!t) return

    setCommentBusy(true)
    try {
      const { error } = await supabase.from('comments').insert({
        listing_id: listingId,
        author_id: user.id,
        text: t,
      })
      if (error) throw error

      await fetchComments(listingId)
    } catch (e) {
      alert(e.message || String(e))
    } finally {
      setCommentBusy(false)
    }
  }

  async function deleteComment(commentId) {
    if (!user) return
    const ok = confirm('刪除這則留言？')
    if (!ok) return

    setCommentBusy(true)
    try {
      // 只刪自己的留言（也要搭配 DB RLS）
      const { error } = await supabase
        .from('comments')
        .delete()
        .eq('id', commentId)
        .eq('author_id', user.id)

      if (error) throw error

      await fetchComments(selectedId)
    } catch (e) {
      alert(e.message || String(e))
    } finally {
      setCommentBusy(false)
    }
  }

  // 初次載入 listings
  useEffect(() => {
    fetchListings()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 打開商品詳情時載入留言
  useEffect(() => {
    if (!selectedId) return
    fetchComments(selectedId)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedId])

  // ====== UI ======
  const editingItem = useMemo(
    () => items.find((x) => x.id === editingId) || null,
    [items, editingId]
  )

  return (
    <Shell
      go={go}
      title="二手交易區"
      subtitle="正式版：資料從 Supabase 讀寫，上架/編輯/刪除/留言都會保存。"
    >
      <div className="toolbar toolbarRow">
        <button className="btn btnGhost" onClick={() => go('home')}>
          ← 回主畫面
        </button>

        <input
          className="control controlGrow"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="搜尋商品（title/tag/size）"
        />

        <button className="btn btnGhost" onClick={() => {/* 重新整理 */}}>
          重新整理
        </button>

        <button className="btn btnPrimary" onClick={() => setModalOpen(true)}>
          ＋ 上架
        </button>
      </div>

      {error && (
        <div style={{ marginTop: 10, padding: 10, border: '1px solid rgba(139,46,46,.35)', borderRadius: 12 }}>
          <strong style={{ color: '#8b2e2e' }}>Error：</strong> {error}
        </div>
      )}

      {loading ? (
        <div style={{ marginTop: 16, opacity: 0.75 }}>載入中...</div>
      ) : (
        <div className="grid" style={{ marginTop: 14 }}>
          {filtered.map((p) => (
            <ProductCard
              key={p.id}
              item={p}
              isMine={p.seller_id === user?.id}
              onOpen={() => setSelectedId(p.id)}
              onEdit={() => {
                setEditingId(p.id)
                setModalOpen(true)
              }}
              onDelete={() => deleteListing(p.id)}
              onSetStatus={(status) => setStatus(p.id, status)}   // 加這行
            />
          ))}
        </div>
      )}

      {/* 新增/編輯共用 Modal */}
      {modalOpen && (
        <ProductModal
          mode={editingItem ? 'edit' : 'add'}
          initial={editingItem}
          onClose={() => {
            setModalOpen(false)
            setEditingId(null)
          }}
          onSubmit={async (form) => {
            if (editingItem) await updateListing(editingItem.id, form)
            else await createListing(form)

            setModalOpen(false)
            setEditingId(null)
          }}
        />
      )}

      {/* 商品詳情（留言區） */}
      {selectedItem && (
        <ProductDetailModal
          item={selectedItem}
          user={user}
          comments={comments}
          loading={commentsLoading}
          busy={commentBusy}
          onClose={() => setSelectedId(null)}
          onAddComment={(text) => addComment(selectedItem.id, text)}
          onDeleteComment={deleteComment}
        />
      )}
    </Shell>
  )
}

/* ======================
   ProductCard：交易區卡片
====================== */
function ProductCard({ item, isMine, onOpen, onEdit, onDelete, onSetStatus }) {
  return (
    <div className="card">
      <img className="cardImg" alt={item.title} src={item.image} />

      {/* 右上角：查看 / 編輯 / 刪除 */}
      <div className="cardActions">
        <button type="button" className="iconBtn" onClick={onOpen} title="查看">
          View
        </button>

        {isMine && (
          <>
            <button type="button" className="iconBtn" onClick={onEdit} title="編輯">
              Edit
            </button>
            <button
              type="button"
              className="iconBtn danger"
              onClick={onDelete}
              title="刪除"
            >
              Delete
            </button>
          </>
        )}
      </div>

      <div className="cardBody">
        <div className="cardTopRow">
          <p className="cardTitle">{item.title}</p>
          <span className="badge">{item.tag}</span>
        </div>

        <div className="meta">
          <span>尺寸：{item.size}</span>
          <span>狀態：{item.condition}</span>
          {/* 顯示目前上架狀態（available/reserved/sold/hidden） */}
          <span>上架：{item.status || 'available'}</span>
        </div>

        <div className="priceRow">
          <span className="price">NT$ {item.price}</span>
          <button type="button" className="btn btnGhost" onClick={onOpen}>
            查看 / 留言
          </button>
        </div>

        {/* 只有自己的商品才顯示：狀態切換 */}
        {isMine && (
          <div className="statusRow">
            <button
              type="button"
              className={`chip ${item.status === 'available' ? 'active' : ''}`}
              onClick={() => onSetStatus?.('available')}
            >
              上架中
            </button>
            <button
              type="button"
              className={`chip ${item.status === 'reserved' ? 'active' : ''}`}
              onClick={() => onSetStatus?.('reserved')}
            >
              保留
            </button>
            <button
              type="button"
              className={`chip ${item.status === 'sold' ? 'active' : ''}`}
              onClick={() => onSetStatus?.('sold')}
            >
              已售
            </button>
            <button
              type="button"
              className={`chip ${item.status === 'hidden' ? 'active' : ''}`}
              onClick={() => onSetStatus?.('hidden')}
            >
              隱藏
            </button>
          </div>
        )}
      </div>
    </div>
  )
}

/* ======================
   ProductModal：上架/編輯表單
   - onSubmit 回傳：{title, price, size, condition, tag, imageUrl, file}
   - 注意：不要把 blob: preview 當成永久圖片存進 DB
====================== */
function ProductModal({ mode, initial, onClose, onSubmit }) {
  const isEdit = mode === 'edit'

  const [title, setTitle] = useState(initial?.title ?? '')
  const [price, setPrice] = useState(initial?.price ?? 300)
  const [size, setSize] = useState(initial?.size ?? 'M')
  const [condition, setCondition] = useState(initial?.condition ?? '9成新')
  const [tag, setTag] = useState(initial?.tag ?? '新上架')

  const [imageUrl, setImageUrl] = useState(initial?.image ?? '')
  const [preview, setPreview] = useState('')
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

  const previewSrc = preview || imageUrl

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
              <label>上傳商品照片（可選）</label>
              <input type="file" accept="image/*" onChange={handleFile} />
              {previewSrc && <img className="previewImg" alt="preview" src={previewSrc} />}
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
              title,
              price,
              size,
              condition,
              tag,
              imageUrl, // 永久 URL（或空）
              file,     // 若有選檔，就交給父層上傳到 Storage
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
   ProductDetailModal：商品詳情 + 留言區（Supabase 版）
====================== */
function ProductDetailModal({ item, user, comments, loading, busy, onClose, onAddComment, onDeleteComment }) {
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

          <div>
            <h4 style={{ margin: '0 0 10px 0' }}>留言區</h4>

            {loading ? (
              <div style={{ opacity: 0.7 }}>載入留言中...</div>
            ) : (
              <div style={{ display: 'grid', gap: 10 }}>
                {comments.length === 0 ? (
                  <div style={{ opacity: 0.7, fontSize: 14 }}>目前還沒有留言。</div>
                ) : (
                  comments.map((c) => (
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

                      {/* 只讓自己刪自己的留言（也要 DB RLS 配合） */}
                      {user?.id && c.author_id === user.id && (
                        <div style={{ marginTop: 8 }}>
                          <button className="btn btnGhost" disabled={busy} onClick={() => onDeleteComment(c.id)}>
                            刪除留言
                          </button>
                        </div>
                      )}
                    </div>
                  ))
                )}
              </div>
            )}

            <div className="toolbar" style={{ marginTop: 12 }}>
              <input
                style={{ flex: 1 }}
                value={text}
                onChange={(e) => setText(e.target.value)}
                placeholder={user ? "輸入留言（例如：請問可小議嗎？）" : "請先登入才能留言"}
                disabled={!user || busy}
              />
              <button className="btn btnPrimary" onClick={submit} disabled={!user || busy}>
                送出
              </button>
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


