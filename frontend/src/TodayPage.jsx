import { useEffect, useMemo, useState } from 'react'
import { supabase } from './supabaseClient.js'
import Shell from './Shell.jsx'

function itemImage(it) {
  return it?.image_url || it?.image || "https://images.unsplash.com/photo-1520975958225-8d56346d1b60?auto=format&fit=crop&w=1200&q=60"
}

export default function TodayPage({ go, user }) {
  // ====== 1. 衣櫃資料讀取 (保持不變) ======
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

  // ====== 2. 表單狀態 (已刪除不必要的欄位) ======
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')
  
  // 新增：用來存儲 AI 辨識出的結果
  const [prediction, setPrediction] = useState(null) // { category: 'jeans', color: 'blue' }

  function handleFile(e) {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setPrediction(null) // 重選圖片時，清空舊的辨識結果
    setResult(null)     // 清空舊的建議
  }

  useEffect(() => {
    return () => {
      if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview)
    }
  }, [preview])

  // ====== 3. AI 分析邏輯 ======
  const [busy, setBusy] = useState(false)
  const [statusText, setStatusText] = useState('') // 用來顯示目前 AI 做到哪一步
  const [result, setResult] = useState(null)

  const closetCount = closet.length

  const topSimilar = useMemo(() => {
    if (!result?.top) return []
    return result.top
  }, [result])

  /** * 🚀 核心功能：
   * 1. 先辨識 (predict_type)
   * 2. 再比對 (compare_url) - 沿用不卡頓邏輯
   */
  async function analyzeWithAI() {
    if (!user?.id) return alert('請先登入才能分析')
    if (!closetCount) return alert('你的衣櫃目前是空的，無法進行比對')
    if (!file) return alert('請上傳一張圖片')

    setBusy(true)
    setResult(null)
    setPrediction(null)
    
    try {
      // --- Phase 1: 辨識衣物類型與顏色 ---
      setStatusText('🔍 AI 正在辨識衣物類型與顏色...')
      
      const formData = new FormData()
      formData.append('file', file)

      // 呼叫後端 model_weights.pth 進行辨識
      const predRes = await fetch('http://127.0.0.1:8000/predict_type', {
        method: 'POST',
        body: formData
      })
      
      if (!predRes.ok) throw new Error('分類模型連線失敗')
      const predData = await predRes.json()
      
      // 取得辨識結果
      const aiCategory = predData.category  // 例如 "Jeans"
      const aiColor = predData.color        // 例如 "Blue"
      
      setPrediction({ category: aiCategory, color: aiColor })
      setStatusText(`✅ 辨識完成！這是一件 ${aiColor} 的 ${aiCategory}`)

      // --- Phase 2: 篩選衣櫃 (只比對同類別) ---
      // 注意：這裡直接使用 AI 辨識出的 aiCategory 來過濾
      let targetItems = closet.filter(c => 
        c.category && c.category.toLowerCase() === aiCategory.toLowerCase()
      )

      if (targetItems.length === 0) {
        // 如果衣櫃裡完全沒有這類衣服，直接給結果
        setResult({
          decision: '值得入手 ✨',
          maxSim: 0,
          reasons: [`你的衣櫃裡完全沒有 ${aiCategory}，這會是你的第一件！`],
          top: []
        })
        setBusy(false)
        return
      }

      setStatusText(`📂 正在衣櫃中搜尋 ${targetItems.length} 件同類衣物...`)

      // --- Phase 3: 相似度比對 (沿用你指定的原始邏輯) ---
      const comparisonPromises = targetItems.map(async (item) => {
        try {
          const compareData = new FormData()
          compareData.append('file1', file)
          compareData.append('url2', itemImage(item)) // 傳網址給後端下載，防止卡頓

          const res = await fetch('http://127.0.0.1:8000/compare_url', {
            method: 'POST',
            body: compareData
          })
          
          if (!res.ok) throw new Error('比對 API 錯誤')
          
          const data = await res.json()
          const simScore = data.similarity / 100 

          return { ...item, sim: simScore }
        } catch (err) {
          console.error("比對失敗:", item.title, err)
          return { ...item, sim: 0 }
        }
      })

      const results = await Promise.all(comparisonPromises)
      results.sort((a, b) => b.sim - a.sim)

      // --- Phase 4: 決策邏輯 (保持不變) ---
      const maxSim = results[0]?.sim ?? 0
      const top = results.slice(0, 3)

      let decision = '可以買 ✅'
      if (maxSim >= 0.80) decision = '千萬不要買 ⛔'
      else if (maxSim >= 0.50) decision = '考慮一下 ⚠️'

      const reasons = []
      if (maxSim >= 0.80) reasons.push(`AI 發現衣櫃裡有幾乎一模一樣的 ${aiCategory}！`)
      else if (maxSim >= 0.50) reasons.push('風格或版型高度雷同，可能會重複穿搭')
      else if (maxSim < 0.30) reasons.push(`這件 ${aiCategory} 風格很獨特，是你衣櫃裡少見的款式`)
      else reasons.push('有些微相似，視搭配需求而定')

      // 穿著頻率判斷
      const best = top[0]
      if (best && maxSim > 0.5) {
        if ((best.worn ?? 0) <= 1) reasons.push(`相似度最高的「${best.title}」你幾乎沒穿過！`)
        else reasons.push(`不過相似度最高的的那件「${best.title}」你很常穿，買這件當替換或許不錯`)
      }

      setResult({ decision, maxSim, reasons, top })

    } catch (err) {
      console.error(err)
      alert("AI 分析發生錯誤，請確認後端是否已開啟？")
    } finally {
      setBusy(false)
      // 稍微延遲清除狀態文字，讓使用者看得到「辨識完成」
      if (!result) setStatusText('')
    }
  }

  return (
    <Shell
      go={go}
      title="智慧購物助手"
      subtitle="上傳你想購買的衣服，AI 掃描衣櫃並檢視你是否有類似風格的衣物。"
    >
      <div className="toolbar toolbarRow">
        <button className="btn btnGhost" onClick={() => go('home')}>← 回主畫面</button>
        <div className="spacer" />
        <div style={{ opacity: 0.75, fontSize: 14 }}>
          衣櫃總數：{loadingCloset ? '...' : closetCount}
        </div>
      </div>

      {error && (
        <div style={{ marginTop: 10, padding: 10, border: '1px solid #8b2e2e', borderRadius: 8, color: '#8b2e2e' }}>
          Error: {error}
        </div>
      )}

      {/* ===== 上傳與操作區 ===== */}
      <div className="card" style={{ marginTop: 14 }}>
        <div className="cardBody">
          
          {/* 圖片預覽區 */}
          <div style={{ textAlign: 'center', marginBottom: 20 }}>
            {preview ? (
              <img 
                src={preview} 
                alt="preview" 
                style={{ maxWidth: '100%', maxHeight: 250, borderRadius: 8, objectFit: 'contain' }} 
              />
            ) : (
              <div style={{ height: 150, background: '#f5f5f5', borderRadius: 8, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#999' }}>
                📷 請上傳照片
              </div>
            )}
          </div>

          {/* AI 狀態顯示條 */}
          {(busy || statusText) && (
            <div style={{ 
              marginBottom: 15, 
              padding: '8px 12px', 
              background: busy ? '#e3f2fd' : '#e8f5e9', 
              color: busy ? '#1565c0' : '#2e7d32',
              borderRadius: 6,
              fontSize: 14,
              textAlign: 'center',
              fontWeight: 500
            }}>
              {statusText || '準備就緒'}
            </div>
          )}

          {/* 辨識結果顯示 (如果有) */}
          {prediction && !busy && (
            <div style={{ marginBottom: 15, textAlign: 'center' }}>
              <span className="badge" style={{ fontSize: 14, padding: '6px 12px', background: '#333', color: '#fff' }}>
                AI 辨識結果：{prediction.color} {prediction.category}
              </span>
            </div>
          )}

          <div style={{ marginBottom: 14 }}>
          <label 
            htmlFor="file-upload" 
            className="btn btnPrimary" 
            style={{ 
              width: '100%', 
              display: 'block', 
              textAlign: 'center', 
              cursor: 'pointer',
              boxSizing: 'border-box' 
            }}
          >
            {preview ? '更換照片' : '上傳照片'}
          </label>
          <input 
            id="file-upload" 
            type="file" 
            accept="image/*" 
            onChange={handleFile} 
            style={{ display: 'none' }} 
          />
        </div>

          <div className="toolbar" style={{ marginTop: 14 }}>
            <button
              className="btn btnPrimary"
              disabled={busy || !file || loadingCloset}
              onClick={analyzeWithAI}
              style={{ width: '100%' }} // 讓按鈕滿版
            >
              {busy ? 'AI 思考中...' : '開始分析決策'}
            </button>
          </div>
        </div>
      </div>

      {/* ===== 結果建議區 ===== */}
      {result && (
        <div className="card" style={{ marginTop: 18, border: result.maxSim >= 0.8 ? '2px solid #ef5350' : '1px solid #ddd' }}>
          <div className="cardBody">
            <div className="cardTopRow">
              <p className="cardTitle" style={{ fontSize: 18, color: result.maxSim >= 0.8 ? '#c62828' : '#2e7d32' }}>
                {result.decision}
              </p>
              <span className="badge">
                最高相似度 {Math.round((result.maxSim || 0) * 100)}%
              </span>
            </div>

            <div className="meta" style={{ marginTop: 10 }}>
              {(result.reasons || []).map((r, idx) => (
                <div key={idx} style={{marginBottom: 4}}>• {r}</div>
              ))}
            </div>

            {result.top.length > 0 && (
              <>
                <div style={{ marginTop: 14, fontWeight: 700, fontSize: 14 }}>
                  因為你有這些很像的衣服：
                </div>
                <div className="grid" style={{ marginTop: 10 }}>
                  {topSimilar.map((it) => (
                    <div key={it.id} className="card" style={{ marginBottom: 0 }}>
                      <img className="cardImg" alt={it.title} src={itemImage(it)} />
                      <div className="cardBody">
                        <div className="cardTopRow">
                          <p className="cardTitle" style={{ fontSize: 13 }}>{it.title || '未命名'}</p>
                          <span className="badge" style={{ 
                            background: it.sim > 0.80 ? '#8b2e2e' : '#eee',
                            color: it.sim > 0.80 ? '#fff' : '#333',
                            fontSize: 11
                          }}>
                            {Math.round((it.sim || 0) * 100)}%
                          </span>
                        </div>
                        <div className="meta" style={{ fontSize: 11 }}>
                          穿過 {it.worn ?? 0} 次
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </Shell>
  )
}