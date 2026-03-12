import { useEffect, useMemo, useState } from 'react'
import { supabase } from './supabaseClient.js'
import Shell from './Shell.jsx'

function shortId(id) {
  return id ? id.slice(0, 6) : 'Unknown'
}

function imgOrFallback(url) {
  return url || 'https://images.unsplash.com/photo-1520975958225-8d56346d1b60?auto=format&fit=crop&w=1200&q=60'
}

export default function MyPage({ go, user, openMarket }) {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  //kpi
  const [kpi, setKpi] = useState({
  avoided: 0,
  listedLow: 0,
  soldRate: null,   // 0~1 或 null
  avgWorn: 0,
  medianWorn: 0,
})
  // Seller: 我的商品
  const [myListings, setMyListings] = useState([])
  const [pendingCountMap, setPendingCountMap] = useState({}) // listing_id -> pending count

  // Buyer: 我送出的詢問
  const [myInquiries, setMyInquiries] = useState([])
  const [listingInfoMap, setListingInfoMap] = useState({}) // listing_id -> {title,image_url,seller_id,status}

  const [displayName, setDisplayName] = useState('')
  const [nameLoading, setNameLoading] = useState(true)
  const [nameSaving, setNameSaving] = useState(false)

  useEffect(() => {
    if (!user?.id) return
    let alive = true

    async function loadProfile() {
      setNameLoading(true)

      const { data, error } = await supabase
        .from('profiles')
        .select('display_name')
        .eq('id', user.id)
        .maybeSingle()

      if (!alive) return

      // 沒有 profile 就自動用 email 前綴建一個（避免空白）
      if (!data?.display_name) {
        const fallback = (user.email?.split('@')?.[0] || 'User').slice(0, 20)
        await supabase.from('profiles').upsert({ id: user.id, display_name: fallback })
        setDisplayName(fallback)
      } else {
        setDisplayName(data.display_name)
      }

      setNameLoading(false)
    }

    loadProfile()
    return () => { alive = false }
  }, [user?.id])

  async function saveName() {
    const name = displayName.trim()
    if (!name) return alert('名字不能空白')
    if (name.length > 20) return alert('名字太長（建議 20 字內）')

    setNameSaving(true)
    const { error } = await supabase
      .from('profiles')
      .upsert({ id: user.id, display_name: name })

    setNameSaving(false)
    if (error) alert(error.message)
  }

  async function logout() {
    await supabase.auth.signOut()
    // App.jsx 會自動回到登入頁
  }
  async function loadAll() {
    if (!user?.id) return
    setLoading(true)
    setError('')

    try {
      // 1) 我上架的商品
      const { data: listings, error: lerr } = await supabase
        .from('market_listings')
        .select('id,title,price,status,tag,image_url,created_at')
        .eq('seller_id', user.id)
        .order('created_at', { ascending: false })
      if (lerr) throw lerr
      setMyListings(listings || [])

      // 2) 統計：每個商品 pending 詢問數（只抓我商品的 listing_id）
      const ids = (listings || []).map(x => x.id)
      if (ids.length) {
        const { data: iq, error: iqErr } = await supabase
          .from('inquiries')
          .select('listing_id')
          .in('listing_id', ids)
          .eq('status', 'pending')
        if (iqErr) throw iqErr

        const map = {}
        for (const row of (iq || [])) {
          map[row.listing_id] = (map[row.listing_id] || 0) + 1
        }
        setPendingCountMap(map)
      } else {
        setPendingCountMap({})
      }

      // 3) 我送出的詢問（買家）
      const { data: myIq, error: myIqErr } = await supabase
        .from('inquiries')
        .select('id,listing_id,message,contact,offer_price,status,created_at')
        .eq('buyer_id', user.id)
        .order('created_at', { ascending: false })
      if (myIqErr) throw myIqErr
      setMyInquiries(myIq || [])

      // 4) 把詢問用到的 listing 資訊補齊（title/image/seller/status）
      const listingIds = [...new Set((myIq || []).map(x => x.listing_id))].filter(Boolean)
      if (listingIds.length) {
        const { data: info, error: infoErr } = await supabase
          .from('market_listings')
          .select('id,title,image_url,seller_id,status')
          .in('id', listingIds)
        if (infoErr) throw infoErr

        const m = {}
        for (const r of (info || [])) m[r.id] = r
        setListingInfoMap(m)
        // ===== KPI 計算 =====

        // A) Avoided duplicate buys：勸退次數（看事件表）
        const { count: avoidedCount } = await supabase
        .from('kpi_events')
        .select('id', { count: 'exact', head: true })
        .eq('user_id', user.id)
        .eq('event_type', 'avoided_duplicate')

        // B) Items listed from low-worn：一鍵上架次數（看事件表）
        const { count: listedLowCount } = await supabase
        .from('kpi_events')
        .select('id', { count: 'exact', head: true })
        .eq('user_id', user.id)
        .eq('event_type', 'listed_from_low_worn')

        // C) Sold rate：sold / total（用 market_listings 的 status）
        const { count: totalListings } = await supabase
        .from('market_listings')
        .select('id', { count: 'exact', head: true })
        .eq('seller_id', user.id)
        .neq('status', 'hidden')

        const { count: soldListings } = await supabase
        .from('market_listings')
        .select('id', { count: 'exact', head: true })
        .eq('seller_id', user.id)
        .eq('status', 'sold')

        const soldRate =
        totalListings && totalListings > 0 ? (soldListings || 0) / totalListings : null

        // D) Closet utilization：平均 worn / 中位數 worn（用 closet_items）
        const { data: wornRows } = await supabase
        .from('closet_items')
        .select('worn')
        .eq('user_id', user.id)

        const wornArr = (wornRows || []).map(r => Number(r.worn) || 0).sort((a, b) => a - b)
        const avgWorn = wornArr.length
        ? wornArr.reduce((s, x) => s + x, 0) / wornArr.length
        : 0

        const medianWorn = wornArr.length
        ? (wornArr.length % 2 === 1
            ? wornArr[(wornArr.length - 1) / 2]
            : (wornArr[wornArr.length / 2 - 1] + wornArr[wornArr.length / 2]) / 2)
        : 0

        setKpi({
        avoided: avoidedCount || 0,
        listedLow: listedLowCount || 0,
        soldRate,
        avgWorn,
        medianWorn,
        })
      } else {
        setListingInfoMap({})
      }
    } catch (e) {
      setError(e.message || String(e))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadAll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.id])

  const sellerPendingTotal = useMemo(() => {
    return Object.values(pendingCountMap).reduce((a, b) => a + (b || 0), 0)
  }, [pendingCountMap])

  const buyerUpdates = useMemo(() => {
    // demo：把非 pending 當作「有更新」
    return (myInquiries || []).filter(x => x.status && x.status !== 'pending').length
  }, [myInquiries])

  return (
    <Shell
      go={go}
      title="My Page"
      subtitle="管理你的上架商品與購買詢問進度。"
    >
      <div className="toolbar toolbarRow">
        <button className="btn btnGhost" onClick={() => go('home')}>← 回主畫面</button>
        <div className="spacer" />
        <button className="btn btnGhost" onClick={loadAll}>更新</button>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <div className="cardBody">
          <div className="cardTopRow">
            <p className="cardTitle" style={{ margin: 0 }}>Account</p>
            <span className="badge">{user.email}</span>
          </div>

          <div style={{ marginTop: 12, display: 'grid', gap: 10 }}>
            <div className="field">
              <label>Display name（交易區顯示用）</label>
              <input
                className="control"
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder={nameLoading ? 'Loading...' : '輸入你想顯示的名字'}
                disabled={nameLoading || nameSaving}
              />
            </div>

            <div className="toolbar" style={{ justifyContent: 'flex-start' }}>
              <button className="btn btnPrimary" onClick={saveName} disabled={nameLoading || nameSaving}>
                {nameSaving ? 'Saving...' : '儲存名字'}
              </button>
              <button className="btn btnGhost" onClick={logout}>
                登出
              </button>
            </div>
          </div>
        </div>
      </div>

      <div className="card" style={{ marginTop: 14 }}>
        <div className="cardBody">
            <div className="cardTopRow">
            <p className="cardTitle" style={{ margin: 0 }}>KPI Dashboard</p>
            <span className="badge">Demo</span>
            </div>

            <div className="kpiGrid">
            <div className="kpiTile">
                <div className="kpiValue">{kpi.avoided}</div>
                <div className="kpiLabel">Avoided duplicate buys</div>
            </div>

            <div className="kpiTile">
                <div className="kpiValue">{kpi.listedLow}</div>
                <div className="kpiLabel">Items listed from low-worn</div>
            </div>

            <div className="kpiTile">
                <div className="kpiValue">
                {kpi.soldRate == null ? '—' : `${Math.round(kpi.soldRate * 100)}%`}
                </div>
                <div className="kpiLabel">Sold rate (sold / listed)</div>
            </div>

            <div className="kpiTile">
                <div className="kpiValue">
                {kpi.avgWorn.toFixed(1)} / {kpi.medianWorn}
                </div>
                <div className="kpiLabel">Closet utilization (avg / median worn)</div>
            </div>
            </div>
        </div>
        </div>
      {error && (
        <div style={{ marginTop: 10, padding: 10, border: '1px solid rgba(139,46,46,.35)', borderRadius: 12 }}>
          <strong style={{ color: '#8b2e2e' }}>Error：</strong> {error}
        </div>
      )}

      {loading ? (
        <div style={{ marginTop: 12, opacity: 0.75 }}>載入中...</div>
      ) : (
        <>
          {/* ===== Seller 區 ===== */}
          <div className="card" style={{ marginTop: 14 }}>
            <div className="cardBody">
              <div className="cardTopRow">
                <p className="cardTitle" style={{ margin: 0 }}>我上架的商品（Seller）</p>
                <span className="badge">待回覆 {sellerPendingTotal}</span>
              </div>

              {myListings.length === 0 ? (
                <div style={{ marginTop: 10, opacity: 0.7 }}>你目前沒有上架商品。</div>
              ) : (
                <div className="grid" style={{ marginTop: 12 }}>
                  {myListings.map((x) => (
                    <div key={x.id} className="card">
                      <img className="cardImg" alt={x.title} src={imgOrFallback(x.image_url)} />
                      <div className="cardBody">
                        <div className="cardTopRow">
                          <p className="cardTitle">{x.title}</p>
                          <span className="badge">{x.status || 'available'}</span>
                        </div>
                        <div className="meta">
                          <span>NT$ {x.price}</span>
                          <span>Tag: {x.tag}</span>
                          <span>Pending: {pendingCountMap[x.id] || 0}</span>
                        </div>
                        <div className="toolbar" style={{ marginTop: 10, justifyContent: 'flex-start' }}>
                          <button className="btn btnGhost" onClick={() => openMarket(x.id)}>
                            查看商品
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          {/* ===== Buyer 區 ===== */}
          <div className="card" style={{ marginTop: 18 }}>
            <div className="cardBody">
              <div className="cardTopRow">
                <p className="cardTitle" style={{ margin: 0 }}>我送出的購買詢問（Buyer）</p>
                <span className="badge">進度更新 {buyerUpdates}</span>
              </div>

              {myInquiries.length === 0 ? (
                <div style={{ marginTop: 10, opacity: 0.7 }}>你目前沒有送出任何詢問。</div>
              ) : (
                <div style={{ marginTop: 12, display: 'grid', gap: 10 }}>
                  {myInquiries.map((iq) => {
                    const info = listingInfoMap[iq.listing_id]
                    return (
                      <div
                        key={iq.id}
                        style={{
                          border: '1px solid rgba(74,44,29,0.15)',
                          borderRadius: 12,
                          padding: 12,
                          background: 'rgba(74,44,29,0.02)'
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 10 }}>
                          <strong style={{ fontSize: 14 }}>
                            {info?.title || `Listing ${shortId(iq.listing_id)}`}
                          </strong>
                          <span className="badge">{iq.status}</span>
                        </div>

                        <div className="meta" style={{ marginTop: 6 }}>
                          {info?.seller_id && <span>賣家：{shortId(info.seller_id)}</span>}
                          {iq.offer_price != null && <span>出價：NT$ {iq.offer_price}</span>}
                          {iq.contact && <span>聯絡：{iq.contact}</span>}
                          {info?.status && <span>商品狀態：{info.status}</span>}
                        </div>

                        <div style={{ marginTop: 6 }}>{iq.message}</div>

                        <div className="toolbar" style={{ marginTop: 10, justifyContent: 'flex-start' }}>
                          <button className="btn btnGhost" onClick={() => openMarket(iq.listing_id)}>
                            查看商品
                          </button>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        </>
      )}
    </Shell>
  )
}