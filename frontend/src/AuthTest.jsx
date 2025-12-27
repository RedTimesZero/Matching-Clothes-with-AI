import { useState } from 'react'
import { supabase } from './supabaseClient.js'

export default function AuthTest({ onLogin }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState(null)

  const signUp = async () => {
    setError(null)
    const { error } = await supabase.auth.signUp({
      email,
      password,
    })
    if (error) setError(error.message)
    else alert('註冊成功，請登入')
  }

  const signIn = async () => {
    setError(null)
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    if (error) setError(error.message)
    else onLogin(data.user)
  }

  return (
    <div style={{ maxWidth: 300 }}>
      <h2>Auth Test</h2>

      <input
        placeholder="email"
        value={email}
        onChange={e => setEmail(e.target.value)}
      />
      <br />

      <input
        type="password"
        placeholder="password"
        value={password}
        onChange={e => setPassword(e.target.value)}
      />
      <br />

      <button onClick={signUp}>註冊</button>
      <button onClick={signIn}>登入</button>

      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  )
}
