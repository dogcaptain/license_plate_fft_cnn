import { useState } from 'react'
import axios from 'axios'
import './App.css'

const API = 'http://localhost:8000/api'

function App() {
  const [mode, setMode] = useState('plate') // 'plate' | 'char'
  const [numChars, setNumChars] = useState(7)
  const [loading, setLoading] = useState(false)
  const [preview, setPreview] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [modelInfo, setModelInfo] = useState(null)

  const handleFile = async (file) => {
    if (!file) return
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
    setLoading(true)

    const formData = new FormData()
    formData.append('file', file)

    try {
      let res
      if (mode === 'plate') {
        res = await axios.post(`${API}/predict-plate?num_chars=${numChars}`, formData)
      } else {
        res = await axios.post(`${API}/predict`, formData)
      }
      setResult(res.data)
    } catch (e) {
      setError(e.response?.data?.error || e.message)
    } finally {
      setLoading(false)
    }
  }

  const onDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }

  const onDragOver = (e) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const fetchModelInfo = async () => {
    try {
      const res = await axios.get(`${API}/model/info`)
      setModelInfo(res.data)
    } catch (e) {
      setError('无法获取模型信息')
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>车牌字符识别系统</h1>
        <p className="subtitle">基于 FFT 频域特征的 CNN 字符识别</p>
      </header>

      <div className="controls">
        <div className="mode-switch">
          <button className={mode === 'plate' ? 'active' : ''} onClick={() => { setMode('plate'); setResult(null) }}>
            车牌识别
          </button>
          <button className={mode === 'char' ? 'active' : ''} onClick={() => { setMode('char'); setResult(null) }}>
            单字符识别
          </button>
        </div>
        {mode === 'plate' && (
          <div className="char-count">
            <label>字符数：</label>
            <select value={numChars} onChange={(e) => setNumChars(Number(e.target.value))}>
              <option value={7}>7位（普通车牌）</option>
              <option value={8}>8位（新能源车牌）</option>
            </select>
          </div>
        )}
        <button className="info-btn" onClick={fetchModelInfo}>模型信息</button>
      </div>

      <div
        className="upload-area"
        onDrop={onDrop}
        onDragOver={onDragOver}
        onClick={() => document.getElementById('file-input').click()}
      >
        {preview ? (
          <img src={preview} alt="预览" className="preview-img" />
        ) : (
          <div className="upload-hint">
            <span className="upload-icon">📁</span>
            <p>拖拽图片到此处，或点击上传</p>
            <p className="hint-sub">{mode === 'plate' ? '支持车牌图片' : '支持单个字符图片'}</p>
          </div>
        )}
        <input
          id="file-input"
          type="file"
          accept="image/*"
          style={{ display: 'none' }}
          onChange={(e) => handleFile(e.target.files[0])}
        />
      </div>

      {loading && <div className="loading">识别中...</div>}
      {error && <div className="error">{error}</div>}

      {/* 单字符结果 */}
      {result && mode === 'char' && (
        <div className="result-section">
          <div className="char-result-single">
            <div className="predicted-char-large">{result.character}</div>
            <div className="confidence">置信度：{(result.confidence * 100).toFixed(1)}%</div>
            <div className="images-row">
              <div className="img-box">
                <p>灰度图</p>
                <img src={`data:image/png;base64,${result.gray_image}`} alt="灰度" />
              </div>
              <div className="img-box">
                <p>FFT 高通特征</p>
                <img src={`data:image/png;base64,${result.fft_image}`} alt="FFT" />
              </div>
            </div>
            <div className="top5">
              <p>Top-5 预测：</p>
              {result.top5.map((item, i) => (
                <div key={i} className="top5-bar">
                  <span className="top5-char">{item.char}</span>
                  <div className="bar-bg">
                    <div className="bar-fill" style={{ width: `${item.prob * 100}%` }} />
                  </div>
                  <span className="top5-prob">{(item.prob * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* 车牌结果 */}
      {result && mode === 'plate' && (
        <div className="result-section">
          <div className="plate-number">{result.plate_number}</div>
          <div className="char-grid">
            {result.characters.map((ch, i) => (
              <div key={i} className="char-card">
                <div className="char-card-char">{ch.character}</div>
                <div className="char-card-conf">{(ch.confidence * 100).toFixed(1)}%</div>
                <div className="char-card-images">
                  <img src={`data:image/png;base64,${ch.gray_image}`} alt="灰度" title="灰度图" />
                  <img src={`data:image/png;base64,${ch.fft_image}`} alt="FFT" title="FFT特征" />
                </div>
                <div className="char-card-top5">
                  {ch.top5.slice(0, 3).map((item, j) => (
                    <div key={j} className="mini-bar">
                      <span>{item.char}</span>
                      <div className="bar-bg-sm">
                        <div className="bar-fill" style={{ width: `${item.prob * 100}%` }} />
                      </div>
                      <span>{(item.prob * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 模型信息弹窗 */}
      {modelInfo && (
        <div className="modal-overlay" onClick={() => setModelInfo(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>模型信息</h3>
            <table>
              <tbody>
                <tr><td>模式</td><td>{modelInfo.mode}</td></tr>
                <tr><td>输入通道</td><td>{modelInfo.in_channels}</td></tr>
                <tr><td>类别数</td><td>{modelInfo.num_classes}</td></tr>
                <tr><td>总参数量</td><td>{modelInfo.total_params?.toLocaleString()}</td></tr>
                <tr><td>设备</td><td>{modelInfo.device}</td></tr>
                <tr><td>HPF Sigma</td><td>{modelInfo.hpf_sigma}</td></tr>
                <tr><td>训练轮次</td><td>{modelInfo.checkpoint_epoch}</td></tr>
                <tr><td>验证准确率</td><td>{modelInfo.checkpoint_val_acc ? (modelInfo.checkpoint_val_acc * 100).toFixed(2) + '%' : 'N/A'}</td></tr>
              </tbody>
            </table>
            <button onClick={() => setModelInfo(null)}>关闭</button>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
