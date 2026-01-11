import { useState } from 'react'
import { supabase } from './supabaseClient.js'

export default function AuthTest({ onLogin }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)

  const signUp = async () => {
    setError(null)
    setBusy(true)
    try {
      const { error } = await supabase.auth.signUp({ email, password })
      if (error) setError(error.message)
      else alert('註冊成功，請到信箱確認後再登入（或依你 Supabase 設定）')
    } finally {
      setBusy(false)
    }
  }

  const signIn = async () => {
    setError(null)
    setBusy(true)
    try {
      const { data, error } = await supabase.auth.signInWithPassword({
        email,
        password,
      })
      if (error) setError(error.message)
      else onLogin(data.user)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="authPage">
      <div className="authCenter">
        <div className="authCard">
          <div className="authHeader">
            <div className="authBrand">My Style Closet</div>
            <h2 className="authTitle">Sign in</h2>
            <p className="authSubtitle">登入後才能使用衣櫃 / 智慧購物助手 / 二手交易</p>
          </div>

          <div className="authForm">
            <div className="field">
              <label>Email</label>
              <input
                className="control"
                placeholder="you@example.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                autoComplete="email"
              />
            </div>

            <div className="field">
              <label>Password</label>
              <input
                className="control"
                type="password"
                placeholder="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                autoComplete="current-password"
              />
            </div>

            {error && <div className="authError">{error}</div>}

            <div className="authActions">
              <button className="btn btnGhost" onClick={signUp} disabled={busy}>
                註冊
              </button>
              <button className="btn btnPrimary" onClick={signIn} disabled={busy}>
                登入
              </button>
            </div>

            <div className="authHint">
              Demo：Supabase Auth（之後可換成 Google OAuth / magic link）
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}