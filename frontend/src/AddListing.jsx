import { useState } from 'react'
import { supabase } from './supabaseClient.js'

export default function AddListing({ user, onAdded }) {
  const [title, setTitle] = useState('')
  const [price, setPrice] = useState('')

  const addListing = async () => {
    if (!title) return alert('請輸入商品名稱')

    const { error } = await supabase
      .from('market_listings')
      .insert({
        seller_id: user.id,
        title,
        price: Number(price) || 0,
      })

    if (error) {
      alert('新增失敗：' + error.message)
    } else {
      setTitle('')
      setPrice('')
      onAdded()
    }
  }

  return (
    <div style={{ marginBottom: 20 }}>
      <h3>上架商品（測試）</h3>

      <input
        placeholder="商品名稱"
        value={title}
        onChange={e => setTitle(e.target.value)}
      />
      <br />

      <input
        placeholder="價格"
        value={price}
        onChange={e => setPrice(e.target.value)}
      />
      <br />

      <button onClick={addListing}>上架</button>
    </div>
  )
}
