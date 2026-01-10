import { supabase } from './supabaseClient.js'
import Shell from './Shell.jsx'
const CATEGORY_OPTIONS = [
  "t-shirt",
  "shirt",
  "hoodie",
  "sweater",
  "jacket",
  "jeans",
  "wide pants",
  "pants", 
  "dress",
  "shorts",
  "skirt",
  "other"
];
/* ======================
   Today Page（今日穿搭推薦）
   目前是 demo 頁：
   - 用固定假資料顯示「今日推薦」與「推薦理由」
   - 喜歡/不喜歡按鈕先做 UI，之後可以接：
     1) 模型回饋（like/dislike 記錄）
     2) 重新生成推薦
====================== */
export default function TodayPage({ go, user }) {
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


