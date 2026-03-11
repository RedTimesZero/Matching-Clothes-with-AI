import { useEffect, useState } from 'react'
import './App.css'
import { supabase } from './supabaseClient.js'

import MarketPage from './MarketPage.jsx'
import ClosetPage from './ClosetPage.jsx'
import TodayPage from './TodayPage.jsx'
import MyPage from './MyPage.jsx'
import AuthTest from './AuthTest.jsx'
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

export default function App() {
  const [page, setPage] = useState('home')
  const [marketOpenId, setMarketOpenId] = useState(null)

  function openMarket(listingId) {
    setMarketOpenId(listingId || null)
    setPage('market')
  }
  // 新增：登入狀態
  const [user, setUser] = useState(null)
  const [authLoading, setAuthLoading] = useState(true)

  useEffect(() => {
    // 1) 先讀 session
    supabase.auth.getSession().then(({ data }) => {
      setUser(data.session?.user ?? null)
      setAuthLoading(false)
    })

    // 2) 監聽登入/登出（避免你登入後畫面不更新）
    const { data: listener } = supabase.auth.onAuthStateChange((_event, session) => {
      setUser(session?.user ?? null)
    })

    return () => {
      listener.subscription.unsubscribe()
    }
  }, [])

  // 還在讀登入狀態
  if (authLoading) return <div style={{ padding: 20 }}>Loading...</div>

  // 沒登入就先顯示登入畫面
  if (!user) return <AuthTest onLogin={setUser} />

  // 下面開始：完全保留你原本的 router / UI
  if (page === 'closet') return <ClosetPage go={setPage} user={user} />
  if (page === 'today') return <TodayPage go={setPage} user={user} />
  if (page === 'market') return <MarketPage go={setPage} user={user} initialSelectedId={marketOpenId} />
  if (page === 'mypage') return <MyPage go={setPage} user={user} openMarket={openMarket} />

  return (
    <div className="home">
      <div className="homeInner">
        <TopNav variant="dark" go={setPage} user={user} />

        <div className="heroContent">
          <div className="heroBox">
            <h1 className="heroTitle">Dress smarter.</h1>
            <p className="heroSubtitle">
              AI智慧衣櫃 | 購物建議 | 衣物一鍵上架二手交易平台。
            </p>

            <div className="heroActions">
              <button className="heroCardBtn" onClick={() => setPage('closet')}>
                進入我的衣櫃
              </button>
              <button className="heroCardBtn" onClick={() => setPage('today')}>
                智慧購物助手
              </button>
              <button className="heroCardBtn" onClick={() => setPage('market')}>
                前往二手交易
              </button>
            </div>

            <p style={{ marginTop: 16, opacity: 0.7 }}>
              已登入：{user.email}
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

function TopNav({ variant, go, user }) {
  const isLight = variant === 'light'
  return (
    <div
      className={`navbar ${isLight ? 'navbarLight' : ''}`}
      style={{ color: isLight ? '#4a2c1d' : '#fff' }}
    >
      <div className="brand" onClick={() => go('home')}>
        My Style Closet
      </div>

      <div className="navMenu">
        <button className="navBtn" onClick={() => go('closet')}>我的衣櫃</button>
        <button className="navBtn" onClick={() => go('today')}>智慧購物助手</button>
        <button className="navBtn" onClick={() => go('market')}>二手交易區</button>
        <button className="navBtn" onClick={() => go('mypage')}>My Page</button>
      </div>
    </div>
  )
}
