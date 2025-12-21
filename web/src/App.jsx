import { useState } from 'react'
import './App.css'

function App() {
  const [page, setPage] = useState('home')

  if (page === 'closet') return <ClosetPage onBack={() => setPage('home')} />
  if (page === 'today') return <TodayPage onBack={() => setPage('home')} />
  if (page === 'market') return <MarketPage onBack={() => setPage('home')} />

  return (
    <div className="home">
      <div className="homeInner">
        <div className="navbar">
          <div className="brand" onClick={() => setPage('home')}>
            My Style Closet
          </div>

          <div className="navMenu">
            <button className="navBtn" onClick={() => setPage('closet')}>
              我的衣櫃
            </button>
            <button className="navBtn" onClick={() => setPage('today')}>
              今日穿搭推薦
            </button>
            <button className="navBtn" onClick={() => setPage('market')}>
              二手交易區
            </button>
          </div>
        </div>

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

function ClosetPage({ onBack }) {
  return (
    <div className="page">
      <div className="pageInner">
        <h2 style={{ fontFamily: 'Playfair Display, serif' }}>我的衣櫃</h2>
        <p>這裡之後會接：上傳衣服照片、分類、顏色分析、穿著次數。</p>
        <button className="backBtn" onClick={onBack}>← 回主畫面</button>
      </div>
    </div>
  )
}

function TodayPage({ onBack }) {
  return (
    <div className="page">
      <div className="pageInner">
        <h2 style={{ fontFamily: 'Playfair Display, serif' }}>今日穿搭推薦</h2>
        <p>這裡之後會接：根據衣櫃 + 配色規則/模型給推薦。</p>
        <button className="backBtn" onClick={onBack}>← 回主畫面</button>
      </div>
    </div>
  )
}

function MarketPage({ onBack }) {
  return (
    <div className="page">
      <div className="pageInner">
        <h2 style={{ fontFamily: 'Playfair Display, serif' }}>二手交易區</h2>
        <p>這裡之後會接：上架清單、買家資訊、商品卡片。</p>
        <button className="backBtn" onClick={onBack}>← 回主畫面</button>
      </div>
    </div>
  )
}

export default App
