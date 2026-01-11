import { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Send, Copy, Check, Info, Loader2, FileText, 
  RefreshCw, Code, Eye, X, ChevronDown, ChevronUp 
} from 'lucide-react'

// ============================================================================
// UTILITIES
// ============================================================================

const renderMarkdown = (text) => {
  if (!text) return '';
  let html = text;
  html = html.replace(/^### (.*$)/gim, '<h3 class="text-sm font-bold font-mono text-slate-800 mt-4 mb-2 uppercase tracking-tight">$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2 class="text-base font-bold font-mono text-slate-800 mt-5 mb-2">$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1 class="text-lg font-bold font-mono text-slate-800 mt-6 mb-3">$1</h1>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong class="font-semibold font-mono text-slate-900">$1</strong>');
  html = html.replace(/`(.+?)`/g, '<code class="bg-slate-100 text-slate-900 px-1 py-0.5 rounded text-xs font-mono">$1</code>');
  html = html.replace(/^\s*[-*+]\s+(.*)$/gim, '<li class="ml-4 list-disc text-slate-600 mb-1 font-mono">$1</li>');
  html = html.replace(/\n/g, '<br>');
  return html;
};

const escapeLaTeX = (text) => {
  const replacements = {
    '\\': '\\textbackslash{}',
    '{': '\\{',
    '}': '\\}',
    '$': '\\$',
    '&': '\\&',
    '%': '\\%',
    '#': '\\#',
    '_': '\\_',
    '^': '\\textasciicircum{}',
    '~': '\\textasciitilde{}'
  };
 
  let escaped = text.replace(/[\\{}$&%#_^~]/g, (match) => replacements[match] || match);
  escaped = escaped.replace(/\n/g, '\\\\\n');
 
  return escaped;
};

const LATEX_PREAMBLE = `\\documentclass[12pt]{article}
\\usepackage[utf8]{inputenc}
\\usepackage[a4paper,margin=0.8in,top=0.5in]{geometry}
\\usepackage{hyperref}
\\usepackage{enumitem}
\\usepackage{changepage}
\\hyphenpenalty = 10000
\\usepackage{parskip}
\\usepackage[T1]{fontenc}
\\usepackage{lmodern}
\\usepackage{sectsty}
\\usepackage{setspace}
\\setstretch{1.14}
\\emergencystretch 10pt
\\sectionfont{\\fontsize{13}{0}\\selectfont}
\\raggedbottom
\\begin{document}
\\pagestyle{empty}
`;

const LATEX_POSTAMBLE = `\n\\end{document}`;

const appGuideContent = `# RAGCV Assistant Guide...`;

// ============================================================================
// CUSTOM HOOKS
// ============================================================================

function useLogs(autoRefreshEnabled) {
  const [logs, setLogs] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef(null);

  const fetchLogs = useCallback(async () => {
    try {
      const response = await fetch('http://localhost:8000/logs');
      const data = await response.json();
      setLogs(data.logs || '');
    } catch (err) { console.error('Log fetch failed'); }
  }, []);

  useEffect(() => {
    let interval;
    if (autoRefreshEnabled) {
      fetchLogs();
      interval = setInterval(fetchLogs, 2000);
    }
    return () => clearInterval(interval);
  }, [autoRefreshEnabled, fetchLogs]);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return { logs, isLoading, fetchLogs, scrollRef };
}

function useLatex() {
  const [latexSource, setLatexSource] = useState('');
  const [pdfUrl, setPdfUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasManualEdits, setHasManualEdits] = useState(false);
 
  const timerRef = useRef(null);
  const abortControllerRef = useRef(null);
  const currentPdfUrlRef = useRef('');

  const compileLaTeX = useCallback(async (latexCode) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
   
    setIsLoading(true);
    setError(null);
   
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch('http://localhost:8000/compile_latex', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ latex: latexCode }),
        signal: controller.signal
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || 'LaTeX compilation failed');
      }

      const pdfBlob = await response.blob();
      const newUrl = URL.createObjectURL(pdfBlob);
     
      if (currentPdfUrlRef.current) {
        URL.revokeObjectURL(currentPdfUrlRef.current);
      }
     
      currentPdfUrlRef.current = newUrl;
      setPdfUrl(newUrl);
    } catch (err) {
      if (err.name === 'AbortError') return;
      setError(err.message || "Failed to compile LaTeX");
    } finally {
      setIsLoading(false);
    }
  }, []);

  const generateInitialLatex = useCallback((plainText) => {
    if (hasManualEdits) return;
   
    const escapedContent = escapeLaTeX(plainText);
    const fullLatex = LATEX_PREAMBLE + escapedContent + LATEX_POSTAMBLE;
    setLatexSource(fullLatex);
    setHasManualEdits(false);
    compileLaTeX(fullLatex);
  }, [hasManualEdits, compileLaTeX]);

  const handleSourceEdit = useCallback((newSource) => {
    setLatexSource(newSource);
    setHasManualEdits(true);
   
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
   
    timerRef.current = setTimeout(() => {
      compileLaTeX(newSource);
    }, 1000);
  }, [compileLaTeX]);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      if (abortControllerRef.current) abortControllerRef.current.abort();
      if (currentPdfUrlRef.current) URL.revokeObjectURL(currentPdfUrlRef.current);
    };
  }, []);

  return {
    latexSource,
    pdfUrl,
    isLoading,
    error,
    generateInitialLatex,
    handleSourceEdit
  };
}

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function App() {
  const [activeTab, setActiveTab] = useState('output');
  const [input, setInput] = useState('');
  const [showAbout, setShowAbout] = useState(false);
  const [copied, setCopied] = useState(false);
  const [leftPaneWidth, setLeftPaneWidth] = useState(45);
  const [isResizing, setIsResizing] = useState(false);
  const [isReportCollapsed, setIsReportCollapsed] = useState(false);
  const [showLatexSource, setShowLatexSource] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [logsWrapText, setLogsWrapText] = useState(true);
  
  const mainRef = useRef(null);
  const logsState = useLogs(autoRefresh);
  const latex = useLatex();
  const [queryState, setQueryState] = useState({ isLoading: false, output: '', summary: '', artifacts: [] });

  // Resizing logic
  const startResizing = () => setIsResizing(true);
  const stopResizing = () => setIsResizing(false);
  const resize = useCallback((e) => {
    if (isResizing && mainRef.current) {
      const val = (e.clientX / mainRef.current.offsetWidth) * 100;
      if (val > 20 && val < 80) setLeftPaneWidth(val);
    }
  }, [isResizing]);

  useEffect(() => {
    if (isResizing) {
      window.addEventListener('mousemove', resize);
      window.addEventListener('mouseup', stopResizing);
    }
    return () => {
      window.removeEventListener('mousemove', resize);
      window.removeEventListener('mouseup', stopResizing);
    };
  }, [isResizing, resize]);

  const handleSubmit = async () => {
    if (!input.trim()) return;
    setQueryState(p => ({ ...p, isLoading: true }));
    setActiveTab('logs');
    try {
      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input }),
      });
      const data = await res.json();
      setQueryState({ isLoading: false, output: data.result, summary: data.summary, artifacts: data.artifacts || [] });
      
      // Generate initial LaTeX with proper formatting
      latex.generateInitialLatex(data.result);

      setActiveTab('output');
    } catch (err) { setQueryState(p => ({ ...p, isLoading: false })); }
  };

  const handleCopy = () => {
    const text = activeTab === 'output' ? queryState.output : activeTab === 'logs' ? logsState.logs : latex.latexSource;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className={`flex flex-col h-screen bg-white font-sans ${isResizing ? 'cursor-col-resize select-none' : ''}`}>
      <header className="flex-none h-14 border-b border-slate-200 flex items-center justify-between px-6 bg-white z-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center shadow-lg shadow-indigo-100">
            <FileText className="w-5 h-5 text-white" />
          </div>
          <h1 className="text-sm font-bold text-slate-900 uppercase tracking-tight">RAGCV Assistant</h1>
        </div>
        <button onClick={() => setShowAbout(true)} className="p-2 text-slate-400 hover:text-indigo-600 transition-colors">
          <Info className="w-5 h-5" />
        </button>
      </header>

      <main ref={mainRef} className="flex-grow flex overflow-hidden relative">
        {/* LEFT PANE */}
        <div className="flex flex-col bg-white" style={{ width: `${leftPaneWidth}%` }}>
          <div className="flex-none px-6 py-3 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center">
            <span className="text-xs font-bold uppercase tracking-widest pb-2 pt-1 border-b-2 border-transparent text-slate-400">Job Description</span>
          </div>
          <textarea
            className="flex-grow p-8 outline-none text-sm leading-relaxed text-slate-600 font-mono resize-none"
            placeholder="Paste job requirements..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <div className="p-6 border-t border-slate-100 bg-white">
            <button
              onClick={handleSubmit}
              disabled={queryState.isLoading || !input.trim()}
              className="w-full flex items-center justify-center gap-2 py-4 bg-indigo-600 hover:bg-indigo-700 disabled:bg-slate-200 text-white rounded-xl font-bold transition-all shadow-lg shadow-indigo-100"
            >
              {queryState.isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
              {queryState.isLoading ? 'PROCESSING...' : 'GENERATE DOCUMENTS'}
            </button>
          </div>
        </div>

        {/* RESIZER */}
        <div onMouseDown={startResizing} className="w-1 h-full hover:bg-slate-200 cursor-col-resize bg-slate-100/50 flex items-center justify-center group">
          <div className="w-0.5 h-12 bg-slate-200 group-hover:bg-indigo-400 rounded-full" />
        </div>

        {/* RIGHT PANE */}
        <div className="flex flex-col bg-white" style={{ width: `${100 - leftPaneWidth}%` }}>
          <div className="flex-none px-6 py-3 border-b border-slate-100 bg-slate-50/50 flex justify-between items-center">
            <div className="flex gap-4">
              {['output', 'logs', 'latex'].map(t => (
                <button
                  key={t}
                  onClick={() => setActiveTab(t)}
                  className={`text-xs font-bold uppercase tracking-widest pb-2 pt-1 border-b-2 transition-all ${
                    activeTab === t ? 'border-indigo-600 text-indigo-600' : 'border-transparent text-slate-400 hover:text-slate-600'
                  }`}
                >
                  {t === 'output' ? 'Plain Text' : t === 'logs' ? 'Logs' : 'LaTeX PDF'}
                </button>
              ))}
            </div>
            <div className="flex items-center gap-3">
              {activeTab === 'latex' && latex.pdfUrl && (
                <button onClick={() => setShowLatexSource(!showLatexSource)} className="flex items-center gap-1.5 text-[10px] font-bold text-slate-500 hover:text-indigo-600 transition-colors bg-slate-100 px-2 py-1 rounded">
                  {showLatexSource ? <Eye className="w-3.5 h-3.5" /> : <Code className="w-3.5 h-3.5" />}
                  {showLatexSource ? 'PREVIEW PDF' : 'EDIT SOURCE'}
                </button>
              )}
              {((activeTab === 'output' && queryState.output) ||
                (activeTab === 'logs' && logsState.logs) ||
                (activeTab === 'latex' && latex.latexSource)) && (
                <button onClick={handleCopy} className="flex items-center gap-1.5 text-[10px] font-bold text-indigo-600 hover:text-indigo-800 transition-colors">
                  {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                  {copied ? 'COPIED' : 'COPY'}
                </button>
              )}
            </div>
          </div>

          <div className="flex-grow overflow-hidden relative">
            {activeTab === 'output' && (
              <div className="h-full overflow-y-auto p-8 space-y-8">
                {queryState.summary && (
                  <div className="border border-slate-200 rounded-xl overflow-hidden bg-white shadow-sm">
                    <button 
                      onClick={() => setIsReportCollapsed(!isReportCollapsed)}
                      className="w-full px-5 py-3 flex items-center justify-between bg-slate-50 border-b border-slate-100"
                    >
                      <span className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Job Intelligence Report</span>
                      {isReportCollapsed ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronUp className="w-4 h-4 text-slate-400" />}
                    </button>
                    {!isReportCollapsed && (
                      <div 
                        className="p-6 text-sm text-slate-600 leading-tight prose-cohesive"
                        dangerouslySetInnerHTML={{ __html: renderMarkdown(queryState.summary) }}
                      />
                    )}
                  </div>
                )}
                <div className="space-y-4">
                  <h3 className="text-xs font-bold uppercase tracking-widest pb-2 pt-1 border-b-2 border-transparent text-slate-400">Final Draft</h3>
                  <pre className="text-sm font-mono text-slate-600 whitespace-pre-wrap leading-relaxed">
                    {queryState.output || 'Awaiting generation...'}
                  </pre>
                </div>
              </div>
            )}

            {activeTab === 'logs' && (
              <div className="h-full flex flex-col">
                <div className="flex-none px-6 py-2 bg-slate-100/50 border-b border-slate-200 flex gap-4">
                   <label className="flex items-center gap-2 text-[10px] font-bold text-slate-500">
                     <input type="checkbox" checked={autoRefresh} onChange={e => setAutoRefresh(e.target.checked)} className="rounded border-slate-300 text-indigo-600" /> AUTO-REFRESH
                   </label>
                   <label className="flex items-center gap-2 text-[10px] font-bold text-slate-500">
                     <input type="checkbox" checked={logsWrapText} onChange={e => setLogsWrapText(e.target.checked)} className="rounded border-slate-300 text-indigo-600" /> WRAP
                   </label>
                </div>
                <div ref={logsState.scrollRef} className="flex-grow overflow-auto p-6 bg-slate-200">
                  <pre className={`text-[12px] font-mono leading-5 ${logsWrapText ? 'whitespace-pre-wrap' : 'whitespace-pre'} text-slate-600`}>
                    {logsState.logs || '> Initializing system logs...'}
                  </pre>
                </div>
              </div>
            )}

            {activeTab === 'latex' && (
              <div className="h-full flex flex-col bg-white">
                {latex.isLoading && !latex.pdfUrl ? (
                  <div className="h-full flex flex-col items-center justify-center text-slate-400">
                    <Loader2 className="w-8 h-8 animate-spin mb-2" />
                    <p className="text-sm">Compiling LaTeX...</p>
                  </div>
                ) : latex.error && !latex.pdfUrl ? (
                  <div className="h-full flex flex-col items-center justify-center text-red-500 p-6">
                    <p className="font-mono text-sm mb-2">LaTeX Compilation Error:</p>
                    <p className="font-mono text-xs text-center">{latex.error}</p>
                  </div>
                ) : latex.pdfUrl ? (
                  <>
                    {!showLatexSource ? (
                      <div className="flex-grow relative bg-slate-100">
                        <iframe
                          key={latex.pdfUrl}
                          src={latex.pdfUrl}
                          className="w-full h-full"
                          title="LaTeX PDF Preview"
                        />
                        {latex.isLoading && (
                          <div className="absolute top-4 right-4 bg-white/90 px-3 py-2 rounded-lg shadow-sm flex items-center gap-2">
                            <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-600" />
                            <span className="text-xs text-slate-600">Recompiling...</span>
                          </div>
                        )}
                        {latex.error && (
                          <div className="absolute bottom-4 left-4 right-4 bg-red-50 border border-red-200 px-4 py-2 rounded-lg">
                            <p className="text-xs text-red-700 font-mono">{latex.error}</p>
                          </div>
                        )}
                      </div>
                    ) : (
                      <div className="flex-grow flex flex-col bg-white">
                        <div className="flex-none px-6 py-2 bg-slate-50 border-b border-slate-100 flex justify-between items-center">
                          <span className="text-xs font-mono font-medium text-slate-400 uppercase tracking-wider">
                            LaTeX Source (Editable)
                          </span>
                          {latex.isLoading && (
                            <div className="flex items-center gap-2 text-xs text-slate-500">
                              <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-600" />
                              <span>Recompiling...</span>
                            </div>
                          )}
                        </div>
                        <textarea
                          className="flex-grow w-full p-6 resize-none outline-none text-xs font-mono leading-relaxed text-slate-700 bg-white"
                          value={latex.latexSource}
                          onChange={(e) => latex.handleSourceEdit(e.target.value)}
                          spellCheck="false"
                        />
                        {latex.error && (
                          <div className="flex-none p-3 bg-red-50 border-t border-red-200">
                            <p className="text-xs text-red-700 font-mono">{latex.error}</p>
                          </div>
                        )}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-slate-300 select-none">
                    <div className="w-16 h-16 border-2 border-slate-100 rounded-full flex items-center justify-center mb-4">
                      <FileText className="w-6 h-6 text-slate-200" />
                    </div>
                    <p className="text-sm">LaTeX PDF will appear here after generation</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
      
      {showAbout && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="bg-white w-full max-w-2xl border border-slate-200 shadow-2xl rounded-2xl flex flex-col overflow-hidden animate-in fade-in zoom-in-95">
            <div className="flex-none px-6 py-4 border-b border-slate-200 flex items-center justify-between">
              <h2 className="text-sm font-bold uppercase tracking-widest">Documentation</h2>
              <button onClick={() => setShowAbout(false)} className="text-slate-400 hover:text-slate-600"><X className="w-5 h-5" /></button>
            </div>
            <div className="p-8 overflow-y-auto max-h-[60vh] text-sm leading-relaxed text-slate-600">
              <div dangerouslySetInnerHTML={{ __html: renderMarkdown(appGuideContent) }} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}