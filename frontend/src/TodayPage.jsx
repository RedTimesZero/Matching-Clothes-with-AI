import { useEffect, useMemo, useState } from 'react'
import { supabase } from './supabaseClient.js'
import Shell from './Shell.jsx'

const CATEGORY_OPTIONS = [
  "Capris", "Jackets", "Jeans", "Leggings", "Shirts", "Shorts", "Skirts", 
  "Sweaters", "Sweatshirts", "Track Pants", "Trousers", "Tshirts", "Tunics"
];

const COLOR_OPTIONS = [
  "Black", "Blue", "Red", "White", "Grey Melange", "Pink", "Charcoal", 
  "Navy Blue", "Grey", "Beige", "Yellow", "Brown", "Green", "Purple", 
  "Turquoise Blue", "Olive", "Cream", "Maroon", "Peach", "Teal", "Lavender", 
  "Orange", "Rust", "Magenta", "Nude", "Sea Green", "Mustard", "Multi", 
  "Gold", "Off White", "Tan", "Mauve", "Khaki", "Coffee Brown", "Burgundy", "Lime Green"
];

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = error => reject(error);
  });
}

function itemImage(it) {
  // 只顯示真正的圖片網址，沒圖就回傳空字串
  return it?.image_url || it?.image || ""
}

// ===== Demo 固定輸出（你要的文字/數字都在這裡改）=====
const DEMO_PREDICTION = { color: 'lavender', category: 'tshirts' }

const DEMO_TOP = [
  {
    id: 'demo-1',
    title: '未命名衣服',
    worn: 0,
    sim: 0.92,
    image_url: '/demo-similar.jpg', // 你準備的照片
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
  // ====== 1) 狀態變數 ======
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState('');
  const [closet, setCloset] = useState([]);
  const [loadingCloset, setLoadingCloset] = useState(true);
  const [error, setError] = useState('');
  const [prediction, setPrediction] = useState(null);
  const [busy, setBusy] = useState(false);
  const [statusText, setStatusText] = useState('');
  const [result, setResult] = useState(null);
  const [isEditing, setIsEditing] = useState(false);

  // ====== 2) 照片上傳處理 ======
  function handleFile(e) {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setPrediction(null);
    setIsEditing(false); // 上傳新照片時關閉編輯模式
  }

  // ====== 3) 讀取真實衣櫃資料 ======
  useEffect(() => {
    if (!user?.id) {
      setCloset([]);
      setLoadingCloset(false);
      return;
    }

    let alive = true; // 防止記憶體洩漏
    
    async function loadCloset() {
      setLoadingCloset(true);
      setError('');
      
      try {
        const { data, error: supabaseError } = await supabase
          .from('closet_items')
          .select('id, title, category, color, worn, image_url, created_at')
          .eq('user_id', user.id)
          .order('created_at', { ascending: false });

        if (!alive) return;
        if (supabaseError) throw supabaseError;

        // ✅ 強制過濾掉沒有圖片或 ID 的髒資料
        const cleanData = (data || []).filter(item => item.id && item.image_url);
        
        setCloset(cleanData);
        console.log(`✅ 成功抓取真實衣櫃資料，共 ${cleanData.length} 件`);
      } catch (err) {
        if (alive) setError(err.message);
      } finally {
        if (alive) setLoadingCloset(false);
      }
    }

    loadCloset();

    return () => { alive = false; };
  }, [user?.id]);

  // 清除預覽圖片記憶體
  useEffect(() => {
    return () => {
      if (preview?.startsWith('blob:')) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  // ====== 4) AI 真實分析與比對邏輯 ======
  async function analyzeWithAI() {
    if (!file) return alert('請上傳一張圖片');

    setBusy(true);
    setResult(null);
    setPrediction(null);
    setStatusText('🔍 AI 正在辨識分類...');

    try {
      const formData = new FormData();
      formData.append('file', file);

      // 確保這裡抓到的是 Vercel 設定的網址
      const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';
      
      // 1. 呼叫分類 API (加入超時保護的想法)
      const res = await fetch(`${API_URL}/predict_type`, {
        method: 'POST',
        body: formData,
      });
      
      if (!res.ok) throw new Error(`分類失敗 (HTTP ${res.status})`);
      
      const aiResult = await res.json();
      setPrediction({ category: aiResult.category, color: aiResult.color });
      setStatusText('🔍 正在比對衣櫃相似度...');

      // 2. 準備比對資料
      if (closet.length === 0) {
        setResult({ decision: '衣櫃是空的，想買就買吧！', maxSim: 0, top: [] });
        return;
      }

      const base64Image = await fileToBase64(file);

      // 3. 呼叫相似度 API
      const compRes = await fetch(`${API_URL}/compare_similarity`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_image: base64Image,
          closet_items: closet.map(it => ({ id: it.id, title: it.title, image_url: it.image_url }))
        }),
      });
      
      if (!compRes.ok) throw new Error(`比對失敗 (HTTP ${compRes.status})`);
      
      const compData = await compRes.json();

      // 4. 設定結果 (此處邏輯不變)
      const topMatch = compData.top_matches[0];
      const maxSim = topMatch ? topMatch.similarity : 0;
      
      setResult({
        decision: maxSim >= 0.8 ? '建議不要買 ⛔' : '這件很適合你！ ✅',
        maxSim: maxSim,
        top: compData.top_matches.map(m => {
            const original = closet.find(i => i.id === m.id);
            return { ...original, sim: m.similarity };
        })
      });

    } catch (err) {
      console.error("❌ 流程中斷:", err);
      alert(`分析失敗: ${err.message}\n請確認 Render 後端網址是否正確且已啟動。`);
    } finally {
      setBusy(false);
      setStatusText('');
    }
  }

  const topSimilar = useMemo(() => {
    if (!result?.top) return [];
    return result.top;
  }, [result]);

  const closetCount = closet.length;

  // ====== 5) UI 渲染區段 ======
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

      {/* 上傳與操作區 */}
      <div className="card" style={{ marginTop: 14 }}>
        <div className="cardBody">
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

          {(busy || statusText) && (
            <div style={{
              marginBottom: 15, padding: '8px 12px',
              background: busy ? '#e3f2fd' : '#e8f5e9', color: busy ? '#1565c0' : '#2e7d32',
              borderRadius: 6, fontSize: 14, textAlign: 'center', fontWeight: 500
            }}>
              {statusText || '準備就緒'}
            </div>
          )}

          {/* 辨識結果與人工校正 */}
          {prediction && !busy && (
            <div style={{ marginBottom: 15, textAlign: 'center', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '8px' }}>
              
              {!isEditing ? (
                /* 顯示模式 */
                <>
                  <span className="badge" style={{ fontSize: 14, padding: '6px 12px', background: '#333', color: '#fff' }}>
                    AI 辨識結果：{prediction.color} {prediction.category}
                  </span>
                  <button 
                    className="btn btnGhost" 
                    style={{ fontSize: 12, padding: '4px 8px', color: '#666', border: '1px solid #ccc' }}
                    onClick={() => setIsEditing(true)}
                  >
                    ✏️ 辨識錯誤？手動修改
                  </button>
                </>
              ) : (
                /* 編輯模式 (下拉選單) */
                <div style={{ display: 'flex', gap: '8px', alignItems: 'center', flexWrap: 'wrap', justifyContent: 'center', background: '#f9f9f9', padding: '10px', borderRadius: '8px', border: '1px solid #ddd' }}>
                  
                  {/* 顏色選單 */}
                  <select 
                    value={prediction.color} 
                    onChange={(e) => setPrediction({ ...prediction, color: e.target.value })}
                    style={{ padding: '6px', borderRadius: '4px', border: '1px solid #ccc', fontSize: 14 }}
                  >
                    {COLOR_OPTIONS.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>

                  {/* 分類選單 */}
                  <select 
                    value={prediction.category} 
                    onChange={(e) => setPrediction({ ...prediction, category: e.target.value })}
                    style={{ padding: '6px', borderRadius: '4px', border: '1px solid #ccc', fontSize: 14 }}
                  >
                    {CATEGORY_OPTIONS.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>

                  <button 
                    className="btn btnPrimary" 
                    style={{ fontSize: 12, padding: '6px 12px' }}
                    onClick={() => setIsEditing(false)}
                  >
                    ✅ 儲存修改
                  </button>
                </div>
              )}
            </div>
          )}

          <div style={{ marginBottom: 14 }}>
            <label
              htmlFor="file-upload"
              className="btn btnPrimary"
              style={{ width: '100%', display: 'block', textAlign: 'center', cursor: 'pointer', boxSizing: 'border-box' }}
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

      {/* 結果建議區 */}
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
  );
}
