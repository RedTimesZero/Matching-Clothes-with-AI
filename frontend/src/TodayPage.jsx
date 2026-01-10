import { useEffect, useMemo, useState } from 'react'
import { supabase } from './supabaseClient.js'
import Shell from './Shell.jsx'

/** 建議：跟 ClosetPage 同一份選單，避免前後不一致 */
const CATEGORY_OPTIONS = [
  "blouse","cardigan","coat","dress","hoodie","jacket","jeans","leggings",
  "pants","shirt","shorts","skirt","sweater","t-shirt","top","vest"
]

const COLOR_OPTIONS = [
  "beige","black","blue","brown","burgundy","cream","gold","gray","green","grey",
  "ivory","khaki","maroon","navy","olive","orange","pink","purple","red","rose",
  "silver","tan","white","yellow"
]

function itemImage(it) {
  return it?.image_url || it?.image || "https://images.unsplash.com/photo-1520975958225-8d56346d1b60?auto=format&fit=crop&w=1200&q=60"
}

export default function TodayPage({ go, user }) {
  // ====== closet load ======
  const [closet, setCloset] = useState([])
  const [loadingCloset, setLoadingCloset] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    if (!user?.id) {
      setCloset([])
      setLoadingCloset(false)
      return
    }

    let alive = true
    async function loadCloset() {
      setLoadingCloset(true)
      setError('')

      const { data, error } = await supabase
        .from('closet_items')
        .select('id,title,category,color,worn,image_url,created_at')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false })

      if (!alive) return
      if (error) setError(error.message)
      setCloset(data || [])
      setLoadingCloset(false)
    }

    loadCloset()
    return () => { alive = false }
  }, [user?.id])

  // ====== candidate form ======
  const [title, setTitle] = useState('')
  const [category, setCategory] = useState(CATEGORY_OPTIONS[0])
  const [color, setColor] = useState(COLOR_OPTIONS[0])
  const [imageUrl, setImageUrl] = useState('')
  const [preview, setPreview] = useState('')
  const [file, setFile] = useState(null) // 這是使用者上傳的新衣服檔案

  function handleFile(e) {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
  }

  useEffect(() => {
    return () => {
      if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview)
    }
  }, [preview])

  const previewSrc = preview || imageUrl

  // ====== analysis result ======
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState(null) // { decision, maxSim, reasons[], top[] }

  const closetCount = closet.length

  const topSimilar = useMemo(() => {
    if (!result?.top) return []
    return result.top
  }, [result])

  /** * 🚀 核心功能：呼叫 Python AI 進行比對 
   * 原理：找出衣櫃中同類別的衣服，逐一傳給 Python 後端算相似度
   */
  async function analyzeWithAI() {
    if (!user?.id) return alert('請先登入才能分析')
    if (!closetCount) return alert('你的衣櫃目前是空的，先新增幾件衣服才好比對喔')
    if (!file) return alert('請上傳一張圖片（目前 AI 需要實體圖片檔案才能分析）')

    setBusy(true)
    setResult(null)
    
    try {
      // 1. 優化：只比對「相同類別」的衣服 (避免拿褲子去比外套，浪費時間)
      // 如果該類別沒衣服，就比對全部
      let targetItems = closet.filter(c => c.category === category)
      if (targetItems.length === 0) targetItems = closet

      // 2. 準備比對結果陣列
      const scoredItems = []

      // 3. 逐一呼叫 Python API (使用 Promise.all 加速)
      const comparisonPromises = targetItems.map(async (item) => {
        try {
          // 下載衣櫃裡的這張圖片變成 Blob (因為 Python 需要檔案流)
          const itemImgRes = await fetch(itemImage(item))
          const itemBlob = await itemImgRes.blob()

          // 建立 FormData
          const formData = new FormData()
          formData.append('file1', file)     // 使用者上傳的新衣服
          formData.append('file2', itemBlob) // 衣櫃裡的舊衣服

          // 呼叫 Python 後端
          const res = await fetch('http://127.0.0.1:8000/compare', {
            method: 'POST',
            body: formData
          })
          
          if (!res.ok) throw new Error('API Error')
          
          const data = await res.json()
          // Python 回傳 0-100，我們轉成 0-1 方便前端處理
          const simScore = data.similarity / 100 

          return { ...item, sim: simScore }
        } catch (err) {
          console.error("比對失敗:", item.title, err)
          return { ...item, sim: 0 }
        }
      })

      // 等待所有圖片比對完成
      const results = await Promise.all(comparisonPromises)
      
      // 排序：相似度高 -> 低
      results.sort((a, b) => b.sim - a.sim)

      // 4. 產生決策邏輯 (根據 AI 分數)
      const maxSim = results[0]?.sim ?? 0
      const top = results.slice(0, 3)

      let decision = '可以買 ✅'
      if (maxSim >= 0.60) decision = '千萬不要買 ⛔' // CLIP 模型 85% 其實就非常像了
      else if (maxSim >= 0.45) decision = '考慮一下 ⚠️'

      const reasons = []
      if (maxSim >= 0.60) reasons.push('AI 發現衣櫃裡有幾乎一模一樣的款式！')
      else if (maxSim >= 0.45) reasons.push('風格或版型高度雷同，可能會重複穿搭')
      else if (maxSim < 0.30) reasons.push('你的衣櫃裡完全沒有這種衣服，是很好的新嘗試！')
      else reasons.push('有些微相似，視搭配需求而定')

      // 額外：加上原本的穿著次數判斷
      const best = top[0]
      if (best && maxSim > 0.6) {
        if ((best.worn ?? 0) <= 1) reasons.push(`而且最像的那件「${best.title}」你幾乎沒穿過！`)
        else reasons.push(`不過最像的那件「${best.title}」你很常穿，買這件當替換或許不錯`)
      }

      setResult({ decision, maxSim, reasons, top })

    } catch (err) {
      console.error(err)
      alert("AI 分析發生錯誤，請確認 Python 伺服器 (uvicorn) 有沒有打開？")
    } finally {
      setBusy(false)
    }
  }

  return (
    <Shell
      go={go}
      title="買衣服建議 (AI 版)"
      subtitle="上傳你想買的衣服，AI 會掃描你的衣櫃找出相似款。"
    >
      {/* 工具列：回首頁 */}
      <div className="toolbar toolbarRow">
        <button className="btn btnGhost" onClick={() => go('home')}>← 回主畫面</button>
        <div className="spacer" />
        <div style={{ opacity: 0.75, fontSize: 14 }}>
          衣櫃件數：{loadingCloset ? '讀取中...' : closetCount}
        </div>
      </div>

      {error && (
        <div style={{ marginTop: 10, padding: 10, border: '1px solid rgba(139,46,46,.35)', borderRadius: 12 }}>
          <strong style={{ color: '#8b2e2e' }}>Error：</strong> {error}
        </div>
      )}

      {/* ===== 表單卡 ===== */}
      <div className="card" style={{ marginTop: 14 }}>
        {previewSrc ? (
          <img className="cardImg" alt="candidate" src={previewSrc} />
        ) : (
          <div
            style={{
              height: 180,
              display: 'grid',
              placeItems: 'center',
              background: '#f4f2ef',
              color: 'rgba(74,44,29,0.7)',
              fontSize: 14
            }}
          >
            （請先上傳照片以進行 AI 分析）
          </div>
        )}

        <div className="cardBody">
          <div className="cardTopRow">
            <p className="cardTitle" style={{ margin: 0 }}>輸入想買的衣服</p>
            <span className="badge">AI Ready</span>
          </div>

          <div className="formGrid" style={{ marginTop: 12 }}>
            <div className="field fieldFull">
              <label>上傳照片（AI 分析必填）</label>
              <input type="file" accept="image/*" onChange={handleFile} />
            </div>

            {/* 隱藏：雖然沒用到但為了版面好看保留網址輸入框 */}
            <div className="field fieldFull" style={{display: 'none'}}>
              <label>圖片網址</label>
              <input
                className="control"
                value={imageUrl}
                onChange={(e) => setImageUrl(e.target.value)}
              />
            </div>

            <div className="field">
              <label>名稱（選填）</label>
              <input
                className="control"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="例如：Uniqlo 外套"
              />
            </div>

            <div className="field">
              <label>類別（用於加速篩選）</label>
              <select className="control" value={category} onChange={(e) => setCategory(e.target.value)}>
                {CATEGORY_OPTIONS.map(opt => <option key={opt} value={opt}>{opt}</option>)}
              </select>
            </div>

            <div className="field">
              <label>顏色</label>
              <select className="control" value={color} onChange={(e) => setColor(e.target.value)}>
                {COLOR_OPTIONS.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
            </div>
          </div>

          <div className="toolbar" style={{ marginTop: 14 }}>
            <button
              className="btn btnPrimary"
              disabled={busy || loadingCloset}
              onClick={analyzeWithAI}
            >
              {busy ? 'AI 正在掃描衣櫃...' : '開始分析'}
            </button>

            <button
              className="btn btnGhost"
              onClick={() => {
                setTitle('')
                setCategory(CATEGORY_OPTIONS[0])
                setColor(COLOR_OPTIONS[0])
                setImageUrl('')
                setPreview('')
                setFile(null)
                setResult(null)
              }}
            >
              清除
            </button>
          </div>
        </div>
      </div>

      {/* ===== 結果卡 ===== */}
      {result && (
        <div className="card" style={{ marginTop: 18 }}>
          <div className="cardBody">
            <div className="cardTopRow">
              <p className="cardTitle" style={{ margin: 0 }}>AI 建議：{result.decision}</p>
              <span className="badge">
                最高相似度 {Math.round((result.maxSim || 0) * 100)}%
              </span>
            </div>

            <div className="meta" style={{ marginTop: 10 }}>
              {(result.reasons || []).map((r, idx) => (
                <div key={idx} style={{marginBottom: 4}}>• {r}</div>
              ))}
            </div>

            <div style={{ marginTop: 14, fontWeight: 700 }}>
              衣櫃裡最像的 3 件：
            </div>

            <div className="grid" style={{ marginTop: 10 }}>
              {topSimilar.map((it) => (
                <div key={it.id} className="card">
                  <img className="cardImg" alt={it.title} src={itemImage(it)} />
                  <div className="cardBody">
                    <div className="cardTopRow">
                      <p className="cardTitle">{it.title}</p>
                      {/* 根據分數顯示不同顏色的標籤 */}
                      <span className="badge" style={{ 
                        background: it.sim > 0.60 ? '#8b2e2e' : (it.sim > 0.6 ? '#d97706' : '#eee'),
                        color: it.sim > 0.45 ? '#fff' : '#333'
                      }}>
                        {Math.round((it.sim || 0) * 100)}%
                      </span>
                    </div>
                    <div className="meta">
                      <span>{it.category}</span>
                      <span>穿過 {it.worn ?? 0} 次</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </Shell>
  )
}