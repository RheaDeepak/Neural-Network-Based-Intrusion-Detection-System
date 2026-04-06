import React, { useState } from 'react'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Textarea } from './ui/textarea'

const DEFAULT_SAMPLE = {
  duration: 0,
  protocol_type: 'tcp',
  service: 'http',
  flag: 'SF',
  src_bytes: 0,
  dst_bytes: 0,
  land: 0,
  wrong_fragment: 0,
  urgent: 0,
  hot: 0,
  num_failed_logins: 0,
  logged_in: 0,
  num_compromised: 0,
  root_shell: 0,
  su_attempted: 0,
  num_root: 0,
  num_file_creations: 0,
  num_shells: 0,
  num_access_files: 0,
  num_outbound_cmds: 0,
  is_host_login: 0,
  is_guest_login: 0,
  count: 0,
  srv_count: 0,
  serror_rate: 0,
  srv_serror_rate: 0,
  rerror_rate: 0,
  srv_rerror_rate: 0,
  same_srv_rate: 0,
  diff_srv_rate: 0,
  srv_diff_host_rate: 0,
  dst_host_count: 0,
  dst_host_srv_count: 0,
  dst_host_same_srv_rate: 0,
  dst_host_diff_srv_rate: 0,
  dst_host_same_src_port_rate: 0,
  dst_host_srv_diff_host_rate: 0,
  dst_host_serror_rate: 0,
  dst_host_srv_serror_rate: 0,
  dst_host_rerror_rate: 0,
  dst_host_srv_rerror_rate: 0
}

export default function InputForm({ onPredict }) {
  const [samples, setSamples] = useState([ { ...DEFAULT_SAMPLE } ])
  const [bulkText, setBulkText] = useState('')
  const [bulkError, setBulkError] = useState('')

  const updateField = (idx, key, value) => {
    const copy = [...samples]
    copy[idx][key] = isNaN(Number(value)) ? value : Number(value)
    setSamples(copy)
  }

  const addSample = () => setSamples([...samples, { ...DEFAULT_SAMPLE }])

  const removeSample = (idx) => setSamples(samples.filter((_, i) => i !== idx))

  const submit = (e) => {
    e.preventDefault()
    onPredict(samples)
  }

  const parseCsv = (text) => {
    const lines = text.split(/\r?\n/).filter(l => l.trim().length)
    if (lines.length < 2) return []
    const headers = lines[0].split(',').map(h => h.trim())
    return lines.slice(1).map(line => {
      const values = line.split(',')
      const obj = {}
      headers.forEach((h, i) => {
        const raw = (values[i] ?? '').trim()
        const num = Number(raw)
        obj[h] = Number.isNaN(num) || raw === '' ? raw : num
      })
      return obj
    })
  }

  const applyBulk = () => {
    setBulkError('')
    if (!bulkText.trim()) return
    try {
      let parsed = []
      if (bulkText.trim().startsWith('[') || bulkText.trim().startsWith('{')) {
        const json = JSON.parse(bulkText)
        parsed = Array.isArray(json) ? json : json.samples || []
      } else {
        parsed = parseCsv(bulkText)
      }
      if (!parsed.length) {
        setBulkError('No valid rows found. Ensure JSON is an array of objects or CSV has headers.')
        return
      }
      setSamples(parsed.map(s => ({ ...DEFAULT_SAMPLE, ...s })))
    } catch (err) {
      setBulkError('Could not parse input. Please check JSON/CSV format.')
    }
  }

  const onFileUpload = async (e) => {
    const file = e.target.files?.[0]
    if (!file) return
    const text = await file.text()
    setBulkText(text)
  }

  return (
    <form onSubmit={submit} className="space-y-6">
      {samples.map((s, idx) => (
  <div className="rounded-lg border border-[#E07A5F]/30 bg-[#3D405B]/65 p-4" key={idx}>
          <div className="flex justify-between items-center mb-3">
            <h3 className="font-semibold text-foreground">Sample #{idx+1}</h3>
            <Button type="button" variant="ghost" size="sm" onClick={() => removeSample(idx)}>Remove</Button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <div>
              <label className="text-xs text-muted">protocol_type</label>
              <Input value={s.protocol_type} onChange={e => updateField(idx, 'protocol_type', e.target.value)} />
            </div>
            <div>
              <label className="text-xs text-muted">service</label>
              <Input value={s.service} onChange={e => updateField(idx, 'service', e.target.value)} />
            </div>
            <div>
              <label className="text-xs text-muted">flag</label>
              <Input value={s.flag} onChange={e => updateField(idx, 'flag', e.target.value)} />
            </div>

            <div>
              <label className="text-xs text-muted">duration</label>
              <Input type="number" value={s.duration} onChange={e => updateField(idx, 'duration', e.target.value)} />
            </div>
            <div>
              <label className="text-xs text-muted">src_bytes</label>
              <Input type="number" value={s.src_bytes} onChange={e => updateField(idx, 'src_bytes', e.target.value)} />
            </div>
            <div>
              <label className="text-xs text-muted">dst_bytes</label>
              <Input type="number" value={s.dst_bytes} onChange={e => updateField(idx, 'dst_bytes', e.target.value)} />
            </div>
          </div>
        </div>
      ))}

      <div className="flex justify-end gap-3">
        <Button type="button" variant="outline" onClick={addSample}>Add sample</Button>
        <Button type="submit">Predict</Button>
      </div>

  <div className="rounded-lg border border-[#81B29A]/30 bg-[#2F3148]/70 p-4">
        <h4 className="font-semibold mb-2 text-foreground">Bulk input (CSV or JSON)</h4>
        <p className="text-xs text-muted mb-3">
          Paste a JSON array of objects or CSV with headers. Any missing fields will default to 0.
        </p>
        <Textarea
          placeholder='JSON example: [{"protocol_type":"tcp","service":"http","flag":"SF","duration":0,"src_bytes":181,"dst_bytes":5450}]'
          value={bulkText}
          onChange={(e) => setBulkText(e.target.value)}
        />
        <div className="flex items-center gap-3 mt-3">
          <Input type="file" accept=".csv,.json" onChange={onFileUpload} className="text-xs" />
          <Button type="button" variant="outline" onClick={applyBulk}>Load into form</Button>
        </div>
        {bulkError && <p className="text-xs text-red-400 mt-2">{bulkError}</p>}
      </div>
    </form>
  )
}
