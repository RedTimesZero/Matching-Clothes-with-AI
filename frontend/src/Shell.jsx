// src/components/Shell.jsx
function TopNav({ variant, go }) {
  const isLight = variant === 'light'
  return (
    <div className={`navbar ${isLight ? 'navbarLight' : ''}`} style={{ color: isLight ? '#4a2c1d' : '#fff' }}>
      <div className="brand" onClick={() => go('home')}>My Style Closet</div>
      <div className="navMenu">
        <button className="navBtn" onClick={() => go('closet')}>我的衣櫃</button>
        <button className="navBtn" onClick={() => go('today')}>智慧購物助手</button>
        <button className="navBtn" onClick={() => go('market')}>二手交易區</button>
      </div>
    </div>
  )
}

export default function Shell({ go, title, subtitle, children }) {
  return (
    <div className="shell">
      <TopNav variant="light" go={go} />
      <div className="container">
        <h1 className="pageTitle">{title}</h1>
        <p className="pageSubtitle">{subtitle}</p>
        {children}
      </div>
    </div>
  )
}