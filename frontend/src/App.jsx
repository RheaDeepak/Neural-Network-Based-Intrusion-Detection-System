import React, { useState, useEffect } from 'react'
import axios from 'axios'
import InputForm from './components/InputForm'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './components/ui/card'
import { Pie, Bar } from 'react-chartjs-2'
import { Chart as ChartJS, ArcElement, Tooltip, Legend, BarElement, CategoryScale, LinearScale } from 'chart.js'

ChartJS.register(ArcElement, Tooltip, Legend, BarElement, CategoryScale, LinearScale)

export default function App() {
  const [classes, setClasses] = useState([])
  const [results, setResults] = useState(null)
  const [insights, setInsights] = useState(null)

  useEffect(() => {
    axios.get('http://localhost:8000/classes').then(res => setClasses(res.data.classes)).catch(()=>{})
    axios.get('http://localhost:8000/insights').then(res => setInsights(res.data)).catch(()=>{})
  }, [])

  const handlePredict = async (samples) => {
    try {
      const res = await axios.post('http://localhost:8000/predict', { samples })
      setResults(res.data.results)
    } catch (e) {
      alert('Error contacting backend. Make sure it is running on http://localhost:8000')
    }
  }

  const pieData = () => {
    if (!results) return null
    const counts = {}
    results.forEach(r => { counts[r.predicted_class] = (counts[r.predicted_class] || 0) + 1 })
    return {
      labels: Object.keys(counts),
      datasets: [{
        data: Object.values(counts),
        backgroundColor: ['#E07A5F', '#81B29A', '#F2CC8F', '#F4F1DE', '#3D405B'],
        borderColor: '#2F3148',
        borderWidth: 2
      }]
    }
  }

  const insightsBar = insights ? {
    labels: Object.keys(insights.label_distribution || {}),
    datasets: [{
      label: 'Samples',
      data: Object.values(insights.label_distribution || {}),
      backgroundColor: ['#E07A5F', '#81B29A', '#F2CC8F', '#F4F1DE', '#3D405B']
    }]
  } : null

  return (
    <div className="min-h-screen bg-gradient-to-b from-[#3D405B] via-[#343752] to-[#2F3148]">
      <div className="max-w-6xl mx-auto px-6 py-10">
        <header className="mb-10">
          <div className="inline-flex items-center gap-2 rounded-full border border-[#F2CC8F]/40 bg-[#E07A5F]/25 px-3 py-1 text-xs text-[#F4F1DE] shadow-glow">
            Neural intrusion detection
          </div>
          <h1 className="text-4xl font-semibold mt-4 text-foreground bg-gradient-to-r from-[#F4F1DE] via-[#F2CC8F] to-[#81B29A] bg-clip-text text-transparent">
            Intrusion Detection Command Center
          </h1>
          <p className="text-muted mt-2 max-w-2xl">
            Upload or enter samples to classify network activity. Explore model insights and dataset patterns in one place.
          </p>
        </header>

        <section className="grid md:grid-cols-3 gap-6 mb-8">
          <Card className="md:col-span-2 shadow-glow border-[#E07A5F]/35 bg-[#3D405B]/70 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Predict from new samples</CardTitle>
              <CardDescription>Fill the form or paste CSV/JSON to batch classify traffic.</CardDescription>
            </CardHeader>
            <CardContent>
              <InputForm onPredict={handlePredict} classes={classes} />
            </CardContent>
          </Card>

          <Card className="border-[#81B29A]/35 bg-[#2F3148]/80 shadow-glowCyan backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Model insights</CardTitle>
              <CardDescription>Quick stats from the training dataset and ANN.</CardDescription>
            </CardHeader>
            <CardContent>
              {!insights && <p className="text-sm text-muted">Loading insights…</p>}
              {insights && (
                <div className="space-y-3 text-sm text-muted">
                  <div className="flex items-center justify-between">
                    <span>Total samples</span>
                    <span className="text-foreground font-semibold">{insights.total_samples}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Features</span>
                    <span className="text-foreground font-semibold">{insights.num_features}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Classes</span>
                    <span className="text-foreground font-semibold">{insights.num_classes}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span>Model params</span>
                    <span className="text-foreground font-semibold">{insights.model_params.toLocaleString()}</span>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </section>

        {results && (
          <Card className="mb-8 border-[#F2CC8F]/35 bg-[#2F3148]/80 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Predictions</CardTitle>
              <CardDescription>Summary table and probability breakdown.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid xl:grid-cols-[1.3fr_1fr] gap-6">
                <div className="overflow-auto rounded-lg border border-white/10">
                  <table className="w-full text-sm">
                    <thead className="bg-[#E07A5F]/20 text-[#F4F1DE]">
                      <tr>
                        <th className="px-3 py-2 text-left">#</th>
                        <th className="px-3 py-2 text-left">Predicted class</th>
                        <th className="px-3 py-2 text-left">Confidence</th>
                        <th className="px-3 py-2 text-left">Protocol</th>
                        <th className="px-3 py-2 text-left">Service</th>
                        <th className="px-3 py-2 text-left">Flag</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-white/5">
                      {results.map((row, idx) => {
                        const probs = row.probabilities || {}
                        const best = probs[row.predicted_class] ?? 0
                        return (
                          <tr key={idx} className="hover:bg-[#81B29A]/15">
                            <td className="px-3 py-2 text-muted">{idx + 1}</td>
                            <td className="px-3 py-2 font-semibold text-foreground">{row.predicted_class}</td>
                            <td className="px-3 py-2 text-muted">{(best * 100).toFixed(2)}%</td>
                            <td className="px-3 py-2 text-muted">{row.input?.protocol_type ?? '-'}</td>
                            <td className="px-3 py-2 text-muted">{row.input?.service ?? '-'}</td>
                            <td className="px-3 py-2 text-muted">{row.input?.flag ?? '-'}</td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="rounded-lg border border-[#81B29A]/35 bg-[#3D405B]/70 p-4">
                  {pieData() && <Pie data={pieData()} />}
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {insightsBar && (
          <Card className="border-[#F2CC8F]/35 bg-[#3D405B]/70 backdrop-blur-sm">
            <CardHeader>
              <CardTitle>Dataset label distribution</CardTitle>
              <CardDescription>Class balance across the combined dataset.</CardDescription>
            </CardHeader>
            <CardContent>
              <Bar data={insightsBar} />
            </CardContent>
          </Card>
        )}

        <footer className="mt-10 text-muted text-sm">
          Backend: FastAPI. Frontend: React + Vite + shadcn/ui.
        </footer>
      </div>
    </div>
  )
}
