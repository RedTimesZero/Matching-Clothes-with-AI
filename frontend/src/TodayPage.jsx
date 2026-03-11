import { useEffect, useMemo, useState } from 'react'
import { supabase } from './supabaseClient.js'
import Shell from './Shell.jsx'

function itemImage(it) {
  return (
    it?.image_url ||
    it?.image ||
    "https://images.unsplash.com/photo-1520975958225-8d56346d1b60?auto=format&fit=crop&w=1200&q=60"
  )
}

// ===== Demo 固定輸出（你要的文字/數字都在這裡改）=====
const DEMO_PREDICTION = { color: 'lavender', category: 'tshirts' }

const DEMO_TOP = [
  {
    id: 'demo-1',
    title: '未命名衣服',
    worn: 0,
    sim: 0.92,
    image_url: '/demo-similar.jpg', // ✅ 你準備的照片
  },
  {
    id: 'demo-2',
    title: '未命名衣服',
    worn: 0,
    sim: 0.81,
    // 可選：如果你有第二張 demo 圖，放 public/demo-similar-2.jpg
    image_url: '/demo-similar-2.jpg',
  },
]

export default function TodayPage({ go, user }) {
  // ====== 1) 衣櫃資料讀取（保留不動；只是 demo 不會拿來判斷） ======
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

  // ====== 2) 上傳狀態 ======
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState('')

  // Demo：固定的「AI 辨識結果」與「建議結果」
  const [prediction, setPrediction] = useState(null)
  const [busy, setBusy] = useState(false)
  const [statusText, setStatusText] = useState('')
  const [result, setResult] = useState(null)

  function handleFile(e) {
    const f = e.target.files?.[0]
    if (!f) return
    setFile(f)
    setPreview(URL.createObjectURL(f))
    setPrediction(null)
    setResult(null)
    setStatusText('')
  }

  useEffect(() => {
    return () => {
      if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview)
    }
  }, [preview])

  // Demo 固定 topSimilar
  const topSimilar = useMemo(() => {
    if (!result?.top) return []
    return result.top
  }, [result])

  // ====== 3) Demo 分析：不呼叫後端，不判斷，直接固定輸出 ======
  async function analyzeWithAI() {
    if (!file) return alert('請上傳一張圖片')

    setBusy(true)
    setResult(null)
    setPrediction(null)

    try {
      // 做一點點「假 loading」，看起來更像 AI 在跑（可刪）
      setStatusText('🔍 AI 正在辨識衣物類型與顏色...')
      await new Promise(r => setTimeout(r, 500))

      setPrediction(DEMO_PREDICTION)
      setStatusText('✅ 辨識完成！')

      await new Promise(r => setTimeout(r, 300))

      const decision = '千萬不要買 ⛔'
      const maxSim = 0.92

      setResult({
        decision,
        maxSim,
        reasons: [
          `AI 發現衣櫃裡有幾乎一模一樣的 ${DEMO_PREDICTION.category}！`,
          '相似度最高的「未命名衣服」你幾乎沒穿過！',
        ],
        top: DEMO_TOP.map(x => ({
          ...x,
          image_url: x.image_url === '/demo-similar-2.jpg' ? '/demo-similar.jpg' : x.image_url
        })),
      })

      /* ✅ KPI：勸退成功就記一筆 event */
      if (user?.id && maxSim >= 0.8) {
        await supabase.from('kpi_events').insert({
          user_id: user.id,
          event_type: 'avoided_duplicate',
          meta: {
            decision,
            maxSim,
            demo: true,
            category: DEMO_PREDICTION.category,
            color: DEMO_PREDICTION.color,
          },
        })
      }

    setStatusText('')
    } catch (err) {
      console.error(err)
      alert('Demo 分析失敗（理論上不會發生）')
    } finally {
      setBusy(false)
    }
  }

  const closetCount = closet.length

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

          {/* 辨識結果顯示 */}
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
              disabled={busy || !file}
              onClick={analyzeWithAI}
              style={{ width: '100%' }}
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
                <div key={idx} style={{ marginBottom: 4 }}>• {r}</div>
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
