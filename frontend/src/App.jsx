import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Target, Users, Trophy, Star, Lock, ArrowLeft, UploadCloud } from 'lucide-react';
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, AreaChart, Area } from 'recharts';
import Hero3D from './components/Hero3D';


function scoreTier(score) {
  if (score >= 0.75) return { label: 'Strong', class: 'strong' };
  if (score >= 0.60) return { label: 'Moderate', class: 'moderate' };
  return { label: 'Weak', class: 'weak' };
}

const SCORE_LABELS = {
  semantic_similarity: "Semantic",
  skill_overlap: "Skills",
  experience_match: "Experience",
  education_match: "Education",
  title_similarity: "Title",
  location_match: "Location",
  behavioral_signal_score: "Behavior",
};

export default function App() {
  const [viewState, setViewState] = useState(() => {
    const path = window.location.pathname;
    if (path === '/dashboard') return 'dashboard';
    if (path === '/loading') return 'loading';
    if (path === '/error') return 'error';
    return 'landing';
  });
  const [candidates, setCandidates] = useState([]);
  const [error, setError] = useState(null);
  const [selectedCandidate, setSelectedCandidate] = useState(null);
  const [metrics, setMetrics] = useState({ count: 0, avg: 0, top: 0, strong: 0 });
  const [candidateFile, setCandidateFile] = useState("sample_candidates.json");
  const fileInputRef = useRef(null);
  const [dragActive, setDragActive] = useState(false);

  useEffect(() => {
    const pathMap = {
      landing: '/home',
      loading: '/loading',
      dashboard: '/dashboard',
      error: '/error'
    };
    const targetPath = pathMap[viewState] || '/home';
    
    if (window.location.pathname !== targetPath) {
      window.history.pushState(null, '', targetPath);
    }
  }, [viewState]);

  useEffect(() => {
    const handlePopState = () => {
      const path = window.location.pathname;
      if (path === '/dashboard') setViewState('dashboard');
      else if (path === '/loading') setViewState('loading');
      else if (path === '/error') setViewState('error');
      else setViewState('landing');
    };
    
    window.addEventListener('popstate', handlePopState);
    return () => window.removeEventListener('popstate', handlePopState);
  }, []);

  const fetchData = async (filename = candidateFile) => {
    setViewState('loading');
    setError(null);
    try {
      const response = await axios.post('http://127.0.0.1:8000/rank', {
        candidate_file: filename,
        job_description_text: "We are looking for a Machine Learning Engineer with Python and NLP experience. Location: Pune.",
        top_k: 50,
        page_size: 50
      }, { timeout: 60000 }); // 60s timeout for cold start
      
      const data = response.data.results;
      setCandidates(data);
      
      if (data.length > 0) {
        setSelectedCandidate(data[0]);
        
        const avg = data.reduce((sum, c) => sum + c.score, 0) / data.length;
        const strongCount = data.filter(c => c.score >= 0.75).length;
        
        setMetrics({
          count: data.length,
          avg: avg.toFixed(3),
          top: data[0].score.toFixed(3),
          strong: strongCount
        });
      } else {
        setCandidates([]);
        setSelectedCandidate(null);
        setMetrics({ count: 0, avg: 0, top: 0, strong: 0 });
      }
      setViewState('dashboard');
    } catch (err) {
      console.error("Error fetching data:", err);
      setError(err.message || "Failed to load candidates from backend. Ensure FastAPI is running.");
      setViewState('error');
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setViewState('loading');
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post('http://127.0.0.1:8000/upload-candidates', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      const newFilename = response.data.filename;
      setCandidateFile(newFilename);
      await fetchData(newFilename);
    } catch (err) {
      console.error("Upload error:", err);
      alert("Failed to upload file. See console for details.");
      setViewState('landing');
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      // Create a pseudo event object to reuse handleFileUpload
      handleFileUpload({ target: { files: e.dataTransfer.files } });
    }
  };

  if (viewState === 'error') {
    return (
      <div className="landing-container" style={{ justifyContent: 'center', alignItems: 'center' }}>
        <Target size={48} color="#ef4444" style={{ marginBottom: '1rem' }} />
        <h2 style={{ color: '#ef4444' }}>Backend Connection Error</h2>
        <p style={{ color: '#475569', marginTop: '0.5rem' }}>{error}</p>
        <button className="btn-back" style={{ marginTop: '2rem' }} onClick={() => setViewState('landing')}>
          <ArrowLeft size={16} /> Back to Home
        </button>
      </div>
    );
  }

  if (viewState === 'loading') {
    return (
      <div className="landing-container" style={{ justifyContent: 'center', alignItems: 'center' }}>
        <div className="spinner"></div>
        <p style={{ marginTop: '1.5rem', color: '#475569', fontWeight: '500' }}>
          Ranking candidates using AI (this may take up to 45s on first load)...
        </p>
      </div>
    );
  }

  if (viewState === 'landing') {
    return (
      <div className="landing-container">
        <nav className="landing-nav">
          <div className="logo-section">
            <Target size={28} color="#2563eb" />
            <h1>ShortlistAI</h1>
          </div>
        </nav>
        
        <main className="landing-hero">
          <div className="hero-left">
            <h2 className="hero-title">Are your candidates good enough?</h2>
            <p className="hero-subtitle">
              A free and fast AI resume checker doing multiple crucial checks to ensure your candidates' experience, skills, and background are technically compatible with your job requirements. Get the best talent effortlessly.
            </p>
            
            <div 
              className={`upload-box ${dragActive ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input 
                type="file" 
                accept=".csv,.json" 
                style={{ display: 'none' }} 
                ref={fileInputRef}
                onChange={handleFileUpload} 
              />
              <p>Drop your candidate CSV/JSON file here or choose a file.</p>
              <span>CSV & JSON only. Max 10MB file size.</span>
              
              <button className="btn-upload" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}>
                <UploadCloud size={18} /> Upload File
              </button>
              
              <div className="privacy-note">
                <Lock size={12} /> Privacy guaranteed
              </div>
            </div>
            
            {/* For demonstration purposes, allow loading sample data without upload */}
            <div style={{ textAlign: 'center', marginTop: '1.5rem' }}>
              <button 
                onClick={() => fetchData("sample_candidates.json")}
                style={{ background: 'transparent', border: 'none', color: '#2563eb', cursor: 'pointer', textDecoration: 'underline', fontSize: '0.875rem' }}
              >
                Or try with demo data
              </button>
            </div>
          </div>
          
          <div className="hero-right">
            <Hero3D />
            <div className="decorative-dashboard">
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '2rem' }}>
                <div>
                  <div style={{ fontSize: '12px', color: '#64748b', fontWeight: 'bold' }}>TOP CANDIDATE SCORE</div>
                  <div style={{ fontSize: '3rem', fontWeight: '800', color: '#059669', lineHeight: '1' }}>92/100</div>
                  <div style={{ fontSize: '12px', color: '#10b981' }}>Strong Fit</div>
                </div>
                <div style={{ width: '80px', height: '80px', borderRadius: '50%', border: '8px solid #10b981', borderRightColor: '#e2e8f0', transform: 'rotate(45deg)' }}></div>
              </div>
              
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>
                  <span>Semantic Match</span><span>95%</span>
                </div>
                <div style={{ width: '100%', height: '6px', background: '#e2e8f0', borderRadius: '3px' }}>
                  <div style={{ width: '95%', height: '100%', background: '#10b981', borderRadius: '3px' }}></div>
                </div>
              </div>
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>
                  <span>Skill Overlap</span><span>85%</span>
                </div>
                <div style={{ width: '100%', height: '6px', background: '#e2e8f0', borderRadius: '3px' }}>
                  <div style={{ width: '85%', height: '100%', background: '#10b981', borderRadius: '3px' }}></div>
                </div>
              </div>
              <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: '#64748b', marginBottom: '4px' }}>
                  <span>Experience</span><span>100%</span>
                </div>
                <div style={{ width: '100%', height: '6px', background: '#e2e8f0', borderRadius: '3px' }}>
                  <div style={{ width: '100%', height: '100%', background: '#10b981', borderRadius: '3px' }}></div>
                </div>
              </div>
            </div>
          </div>
        </main>
      </div>
    );
  }

  // Dashboard View
  const chartData = candidates.slice(0, 50).map(c => ({
    name: c.candidate_id,
    score: Number(c.score.toFixed(3)),
    fill: scoreTier(c.score).class === 'strong' ? '#059669' : scoreTier(c.score).class === 'moderate' ? '#d97706' : '#dc2626'
  })).reverse();

  const distributionData = candidates.map(c => ({
    name: c.candidate_id,
    score: Number(c.score.toFixed(3))
  }));

  const radarData = selectedCandidate ? Object.keys(SCORE_LABELS)
    .filter(key => selectedCandidate[key] !== undefined)
    .map(key => ({
      subject: SCORE_LABELS[key],
      A: selectedCandidate[key],
      fullMark: 1,
    })) : [];

  return (
    <div className="app-container dashboard-view">
      <header className="header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <Target size={36} color="#2563eb" />
          <div>
            <h1>ShortlistAI Dashboard</h1>
            <p>Reviewing file: {candidateFile}</p>
          </div>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button className="btn-back" onClick={() => setViewState('landing')}>
            <ArrowLeft size={16} /> New Upload
          </button>
        </div>
      </header>

      <div className="kpi-grid">
        <div className="card">
          <div className="card-title"><Users size={18} /> Candidates</div>
          <div className="kpi-value">{metrics.count}</div>
        </div>
        <div className="card">
          <div className="card-title"><Star size={18} /> Average Score</div>
          <div className="kpi-value">{metrics.avg}</div>
        </div>
        <div className="card">
          <div className="card-title"><Trophy size={18} /> Top Score</div>
          <div className="kpi-value">{metrics.top}</div>
        </div>
        <div className="card">
          <div className="card-title"><Target size={18} /> Strong Fit</div>
          <div className="kpi-value">{metrics.strong}</div>
        </div>
      </div>

      <div className="main-grid">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="table-container">
            <table>
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Candidate ID</th>
                  <th>Score</th>
                  <th>Tier</th>
                </tr>
              </thead>
              <tbody>
                {candidates.map((c) => (
                  <tr 
                    key={c.candidate_id} 
                    onClick={() => setSelectedCandidate(c)}
                    className={selectedCandidate?.candidate_id === c.candidate_id ? 'selected' : ''}
                  >
                    <td>#{c.rank}</td>
                    <td>{c.candidate_id}</td>
                    <td>{c.score.toFixed(4)}</td>
                    <td>
                      <span className={`tier-badge ${scoreTier(c.score).class}`}>
                        {scoreTier(c.score).label}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          
          <div className="card" style={{ flex: 1, minHeight: '300px', display: 'flex', flexDirection: 'column' }}>
            <h2 style={{ marginBottom: '1rem', fontSize: '1.25rem', flexShrink: 0 }}>Score Distribution Curve</h2>
            <div style={{ flex: 1, minHeight: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={distributionData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="colorScore" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#2563eb" stopOpacity={0.5}/>
                      <stop offset="95%" stopColor="#2563eb" stopOpacity={0}/>
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="name" hide />
                  <YAxis domain={['dataMin - 0.02', 'dataMax + 0.02']} tick={{ fontSize: 11, fill: '#475569' }} axisLine={false} tickLine={false} />
                  <Tooltip />
                  <Area type="monotone" dataKey="score" stroke="#2563eb" strokeWidth={2} fillOpacity={1} fill="url(#colorScore)" />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="sidebar">
          {selectedCandidate && (
            <div className="card">
              <h2 style={{ marginBottom: '1.5rem', fontSize: '1.25rem' }}>
                {selectedCandidate.candidate_id} Details
              </h2>
              
              <div style={{ height: 250, marginBottom: '1.5rem' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <RadarChart cx="50%" cy="50%" outerRadius="70%" data={radarData}>
                    <PolarGrid stroke="#e2e8f0" />
                    <PolarAngleAxis dataKey="subject" tick={{ fill: '#475569', fontSize: 11 }} />
                    <PolarRadiusAxis angle={30} domain={[0, 1]} tick={false} axisLine={false} />
                    <Radar name="Candidate" dataKey="A" stroke="#2563eb" fill="#3b82f6" fillOpacity={0.4} />
                    <Tooltip />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              <div className="reasoning-box">
                {selectedCandidate.reasoning}
              </div>

              <div style={{ marginTop: '2rem' }}>
                <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--text-secondary)' }}>Score Breakdown</h3>
                {Object.keys(SCORE_LABELS).filter(k => selectedCandidate[k] !== undefined).map(key => (
                  <div key={key} style={{ marginBottom: '0.75rem' }}>
                    <div className="progress-label">
                      <span>{SCORE_LABELS[key]}</span>
                      <span>{selectedCandidate[key].toFixed(3)}</span>
                    </div>
                    <div className="progress-bar-bg">
                      <div 
                        className="progress-bar-fill" 
                        style={{ width: `${selectedCandidate[key] * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="card" style={{ height: '400px', display: 'flex', flexDirection: 'column' }}>
            <h2 style={{ marginBottom: '1rem', fontSize: '1.25rem', flexShrink: 0 }}>Top Candidates</h2>
            <div style={{ flex: 1, minHeight: 0 }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                  <XAxis type="number" domain={[0, 1]} hide />
                  <YAxis dataKey="name" type="category" tick={{ fill: '#475569', fontSize: 11 }} axisLine={false} tickLine={false} />
                  <Tooltip cursor={{ fill: 'transparent' }} />
                  <Bar dataKey="score" radius={[0, 4, 4, 0]} barSize={12} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
