import { useState, useEffect, useRef, useCallback } from 'react';
import { Send, Copy, Check, Info, Loader2, FileText, RefreshCw, Code, Eye } from 'lucide-react';
// ============================================================================
// UTILITIES
// ============================================================================
const renderMarkdown = (text) => {
  if (!text) return '';
 
  let html = text;
  html = html.replace(/^### (.*$)/gim, '<h3 class="text-sm font-bold text-indigo-900 mt-4 mb-2">$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2 class="text-base font-bold text-indigo-900 mt-5 mb-2">$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1 class="text-lg font-bold text-indigo-900 mt-6 mb-3">$1</h1>');
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong class="font-semibold text-indigo-950">$1</strong>');
  html = html.replace(/__(.+?)__/g, '<strong class="font-semibold text-indigo-950">$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em class="italic">$1</em>');
  html = html.replace(/_(.+?)_/g, '<em class="italic">$1</em>');
  html = html.replace(/`(.+?)`/g, '<code class="bg-indigo-100 text-indigo-900 px-1 py-0.5 rounded text-xs font-mono">$1</code>');
  html = html.replace(/^\* (.+)$/gim, '<li class="ml-4 mb-1">• $1</li>');
  html = html.replace(/^- (.+)$/gim, '<li class="ml-4 mb-1">• $1</li>');
  html = html.replace(/^\d+\. (.+)$/gim, '<li class="ml-4 mb-1">$1</li>');
  html = html.replace(/\n\n/g, '</p><p class="mb-3">');
  html = html.replace(/\n/g, '<br/>');
 
  if (!html.startsWith('<')) {
    html = '<p class="mb-3">' + html + '</p>';
  }
 
  return html;
};
// Improved LaTeX escaping with newline handling
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
 
  // Escape special characters
  let escaped = text.replace(/[\\{}$&%#_^~]/g, (match) => replacements[match] || match);
 
  // Convert double newlines to paragraph breaks
  escaped = escaped.replace(/\n\n+/g, '\n\n\\par\n');
 
  // Convert single newlines to line breaks
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
// ============================================================================
// CUSTOM HOOKS
// ============================================================================
// Hook for debounced LaTeX compilation
const useLatexCompiler = () => {
  const [latexSource, setLatexSource] = useState('');
  const [pdfUrl, setPdfUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hasManualEdits, setHasManualEdits] = useState(false);
 
  const timerRef = useRef(null);
  const abortControllerRef = useRef(null);
  const currentPdfUrlRef = useRef('');
  const compileLaTeX = useCallback(async (latexCode) => {
    // Cancel previous request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
   
    setIsLoading(true);
    setError(null);
   
    const controller = new AbortController();
    abortControllerRef.current = controller;
    try {
      const response = await fetch('/compile_latex', {
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
     
      // Revoke old URL only when we have a valid replacement
      if (currentPdfUrlRef.current) {
        URL.revokeObjectURL(currentPdfUrlRef.current);
      }
     
      currentPdfUrlRef.current = newUrl;
      setPdfUrl(newUrl);
    } catch (err) {
      if (err.name === 'AbortError') {
        // Request was cancelled, ignore
        return;
      }
      setError(err.message || "Failed to compile LaTeX");
    } finally {
      setIsLoading(false);
    }
  }, []);
  const generateInitialLatex = useCallback((plainText) => {
    if (hasManualEdits) return; // Never overwrite manual edits
   
    const escapedContent = escapeLaTeX(plainText);
    const fullLatex = LATEX_PREAMBLE + escapedContent + LATEX_POSTAMBLE;
    setLatexSource(fullLatex);
    setHasManualEdits(false);
    compileLaTeX(fullLatex);
  }, [hasManualEdits, compileLaTeX]);
  const handleSourceEdit = useCallback((newSource) => {
    setLatexSource(newSource);
    setHasManualEdits(true);
   
    // Clear existing timer
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
   
    // Debounce compilation
    timerRef.current = setTimeout(() => {
      compileLaTeX(newSource);
    }, 1000);
  }, [compileLaTeX]);
  // Cleanup
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
};
// Hook for auto-refreshing logs
const useAutoRefreshLogs = (enabled, interval = 2000) => {
  const [logs, setLogs] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const abortControllerRef = useRef(null);
  const fetchLogs = useCallback(async () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
   
    setIsLoading(true);
    const controller = new AbortController();
    abortControllerRef.current = controller;
   
    try {
      const response = await fetch('/logs', {
        signal: controller.signal
      });
      if (!response.ok) throw new Error(`Failed to fetch logs: ${response.status}`);
      const data = await response.json();
      setLogs(data.logs || 'No logs available');
    } catch (err) {
      if (err.name === 'AbortError') return;
      setLogs(`Error loading logs: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  }, []);
  useEffect(() => {
    if (enabled) {
      fetchLogs();
      const interval_id = setInterval(fetchLogs, interval);
      return () => clearInterval(interval_id);
    }
  }, [enabled, interval, fetchLogs]);
  useEffect(() => {
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);
  return { logs, isLoading, fetchLogs };
};
// ============================================================================
// MAIN COMPONENT
// ============================================================================
function App() {
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [summaryOutput, setSummaryOutput] = useState('');
  const [retrievedArtifacts, setRetrievedArtifacts] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState(null);
  const [showAbout, setShowAbout] = useState(false);
  const [activeTab, setActiveTab] = useState('output');
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [logsWrapText, setLogsWrapText] = useState(true);
  const [showLatexSource, setShowLatexSource] = useState(false);
  const [leftPaneWidth, setLeftPaneWidth] = useState(50);
  const [isResizing, setIsResizing] = useState(false);
 
  const containerRef = useRef(null);
  const queryAbortControllerRef = useRef(null);
  // Custom hooks
  const latex = useLatexCompiler();
  const logsState = useAutoRefreshLogs(autoRefresh && activeTab === 'logs');
  // Manual log fetch on tab switch
  useEffect(() => {
    if (activeTab === 'logs' && !logsState.logs) {
      logsState.fetchLogs();
    }
  }, [activeTab, logsState]);
  const handleSubmit = useCallback(async () => {
    if (!input.trim()) return;
    // Cancel previous request
    if (queryAbortControllerRef.current) {
      queryAbortControllerRef.current.abort();
    }
    setIsLoading(true);
    setError(null);
    setOutput('');
    setSummaryOutput('');
    setRetrievedArtifacts([]);
    const controller = new AbortController();
    queryAbortControllerRef.current = controller;
    try {
      const response = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input }),
        signal: controller.signal
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(errorText || `Server error: ${response.status}`);
      }
      const data = await response.json();
      const resultText = data.result || JSON.stringify(data, null, 2);
      const summaryText = data.summary || '';
      const artifacts = data.retrieved_artifacts || [];
      
      setOutput(resultText);
      setSummaryOutput(summaryText);
      setRetrievedArtifacts(artifacts);
      
      
      if (activeTab === 'logs') {
        logsState.fetchLogs();
      }
    } catch (err) {
      if (err.name === 'AbortError') return;
      setError(err.message || "Failed to connect to the assistant.");
    } finally {
      setIsLoading(false);
    }
  }, [input, activeTab, latex, logsState]);
  const handleCopy = useCallback(() => {
    let textToCopy = '';
    if (activeTab === 'output') textToCopy = output;
    else if (activeTab === 'logs') textToCopy = logsState.logs;
    else if (activeTab === 'latex') textToCopy = latex.latexSource;
   
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [activeTab, output, logsState.logs, latex.latexSource]);
  // Resize handle
  const handleMouseDown = useCallback((e) => {
    setIsResizing(true);
    e.preventDefault();
  }, []);
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing || !containerRef.current) return;
     
      const container = containerRef.current;
      const containerRect = container.getBoundingClientRect();
      const newLeftWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100;
     
      if (newLeftWidth >= 20 && newLeftWidth <= 80) {
        setLeftPaneWidth(newLeftWidth);
      }
    };
    const handleMouseUp = () => {
      setIsResizing(false);
    };
    if (isResizing) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isResizing]);
  // Cleanup abort controller
  useEffect(() => {
    return () => {
      if (queryAbortControllerRef.current) {
        queryAbortControllerRef.current.abort();
      }
    };
  }, []);
  const toggleLatexView = useCallback(() => setShowLatexSource(v => !v), []);
  const toggleAutoRefresh = useCallback((e) => setAutoRefresh(e.target.checked), []);
  const toggleLogsWrap = useCallback((e) => setLogsWrapText(e.target.checked), []);
  return (
    <div className="flex flex-col h-screen overflow-hidden bg-white text-slate-900 font-sans selection:bg-indigo-50 selection:text-indigo-900">
     
      <header className="flex-none h-16 border-b border-slate-200 flex items-center justify-between px-6 bg-white z-10">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 bg-indigo-600 rounded-full"></div>
          <h1 className="font-semibold text-lg tracking-tight">Job Application Assistant</h1>
        </div>
        <button
          onClick={() => setShowAbout(!showAbout)}
          className="text-sm font-medium text-slate-500 hover:text-indigo-600 transition-colors flex items-center gap-2"
        >
          <Info className="w-4 h-4" />
          <span>About</span>
        </button>
      </header>
      <main ref={containerRef} className="flex-grow flex flex-col lg:flex-row overflow-hidden relative">
       
        {/* LEFT PANE */}
        <div
          className="flex flex-col border-r border-slate-200 relative group"
          style={{
            width: window.innerWidth >= 1024 ? `${leftPaneWidth}%` : '100%',
            height: window.innerWidth >= 1024 ? '100%' : '50%'
          }}
        >
          <div className="flex-none px-6 py-3 bg-slate-50/50 border-b border-slate-100 flex justify-between items-center">
            <span className="text-xs font-mono font-medium text-slate-400 uppercase tracking-wider">Input Source</span>
            <span className="text-xs text-slate-400">Paste Job Description</span>
          </div>
         
          <textarea
            className="flex-grow w-full p-6 resize-none outline-none text-sm font-mono leading-relaxed text-slate-800 placeholder:text-slate-300 bg-white"
            placeholder="Paste raw job description text here..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={isLoading}
            spellCheck="false"
          />
          <div className="absolute bottom-6 right-6 z-20">
            <button
              onClick={handleSubmit}
              disabled={isLoading || !input.trim()}
              className={`flex items-center gap-2 px-5 py-3 rounded-full font-medium shadow-sm transition-all duration-200 border
                ${isLoading
                  ? 'bg-white border-slate-200 text-slate-400 cursor-not-allowed'
                  : 'bg-indigo-600 border-indigo-600 text-white hover:bg-indigo-700 hover:shadow-md'
                }`}
            >
              {isLoading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : (
                <Send className="w-4 h-4" />
              )}
              <span>{isLoading ? 'Processing...' : 'Generate'}</span>
            </button>
          </div>
        </div>
        {/* RESIZE HANDLE */}
        <div
          className="hidden lg:block w-1 bg-slate-200 hover:bg-indigo-400 cursor-col-resize relative group transition-colors"
          onMouseDown={handleMouseDown}
        >
          <div className="absolute inset-y-0 -left-1 -right-1" />
          {isResizing && <div className="absolute inset-0 bg-indigo-500" />}
        </div>
        {/* RIGHT PANE */}
        <div
          className="flex flex-col bg-slate-50/30"
          style={{
            width: window.innerWidth >= 1024 ? `${100 - leftPaneWidth}%` : '100%',
            height: window.innerWidth >= 1024 ? '100%' : '50%'
          }}
        >
          <div className="flex-none border-b border-slate-100 bg-slate-50/50">
            <div className="flex items-center justify-between px-6 py-2">
              <div className="flex gap-1">
                <button
                  onClick={() => setActiveTab('output')}
                  className={`px-4 py-2 text-xs font-medium rounded-t transition-colors ${
                    activeTab === 'output'
                      ? 'bg-white text-indigo-600 border-b-2 border-indigo-600'
                      : 'text-slate-500 hover:text-slate-700'
                  }`}
                >
                  Plain Text
                </button>
                <button
                  onClick={() => setActiveTab('logs')}
                  className={`px-4 py-2 text-xs font-medium rounded-t transition-colors ${
                    activeTab === 'logs'
                      ? 'bg-white text-indigo-600 border-b-2 border-indigo-600'
                      : 'text-slate-500 hover:text-slate-700'
                  }`}
                >
                  Logs
                </button>
                <button
                  onClick={() => setActiveTab('latex')}
                  className={`px-4 py-2 text-xs font-medium rounded-t transition-colors ${
                    activeTab === 'latex'
                      ? 'bg-white text-indigo-600 border-b-2 border-indigo-600'
                      : 'text-slate-500 hover:text-slate-700'
                  }`}
                >
                  LaTeX PDF
                </button>
              </div>
             
              <div className="flex items-center gap-2">
                {activeTab === 'logs' && (
                  <>
                    <label className="flex items-center gap-1.5 text-xs text-slate-500 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={autoRefresh}
                        onChange={toggleAutoRefresh}
                        className="w-3 h-3 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                      />
                      Auto-refresh
                    </label>
                    <label className="flex items-center gap-1.5 text-xs text-slate-500 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={logsWrapText}
                        onChange={toggleLogsWrap}
                        className="w-3 h-3 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                      />
                      Wrap text
                    </label>
                    <button
                      onClick={logsState.fetchLogs}
                      disabled={logsState.isLoading}
                      className="flex items-center gap-1 text-xs font-medium text-slate-500 hover:text-indigo-600 transition-colors disabled:opacity-50"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 ${logsState.isLoading ? 'animate-spin' : ''}`} />
                      Refresh
                    </button>
                  </>
                )}
                {activeTab === 'latex' && latex.pdfUrl && (
                  <button
                    onClick={toggleLatexView}
                    className="flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-indigo-600 transition-colors"
                  >
                    {showLatexSource ? <Eye className="w-3.5 h-3.5" /> : <Code className="w-3.5 h-3.5" />}
                    {showLatexSource ? 'View PDF' : 'View Source'}
                  </button>
                )}
                {((activeTab === 'output' && output) ||
                  (activeTab === 'logs' && logsState.logs) ||
                  (activeTab === 'latex' && latex.latexSource)) && (
                  <button
                    onClick={handleCopy}
                    className="flex items-center gap-1.5 text-xs font-medium text-indigo-600 hover:text-indigo-800 transition-colors"
                  >
                    {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
                    {copied ? 'Copied' : activeTab === 'latex' ? 'Copy LaTeX' : 'Copy'}
                  </button>
                )}
              </div>
            </div>
          </div>
          <div className="flex-grow overflow-auto relative">
            {activeTab === 'output' ? (
              <div className="h-full overflow-auto p-6">
                {error ? (
                  <div className="h-full flex flex-col items-center justify-center text-red-500">
                    <p className="font-mono text-sm">Error: {error}</p>
                  </div>
                ) : output ? (
                  <div className="space-y-6">
                    {summaryOutput && (
                      <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-5">
                        <div className="flex items-center gap-2 mb-3">
                          <div className="w-2 h-2 bg-indigo-600 rounded-full"></div>
                          <h3 className="text-xs font-semibold text-indigo-900 uppercase tracking-wider">
                            Summary Analysis
                          </h3>
                        </div>
                        <div
                          className="text-sm text-indigo-900 leading-7 prose prose-sm max-w-none prose-headings:text-indigo-900 prose-strong:text-indigo-950"
                          dangerouslySetInnerHTML={{ __html: renderMarkdown(summaryOutput) }}
                        />
                      </div>
                    )}
                    {/* Retrieved Artifacts Bubble */}
                    {retrievedArtifacts.length > 0 && (
                      <div className="bg-slate-50 border border-slate-300 rounded-lg p-5">
                        <div className="flex items-center gap-2 mb-3">
                          <div className="w-2 h-2 bg-slate-600 rounded-full"></div>
                          <h3 className="text-xs font-semibold text-slate-700 uppercase tracking-wider">
                            Retrieved Artifacts ({retrievedArtifacts.length})
                          </h3>
                        </div>
                        <div className="space-y-4">
                          {retrievedArtifacts.map((artifact, idx) => (
                            <div key={idx} className="border-l-2 border-slate-400 pl-4">
                              <div className="flex items-center gap-3 mb-2">
                                <span className="text-xs font-mono text-slate-500">
                                  Score: {artifact.retrieval_score?.toFixed(3)}
                                </span>
                                <span className="text-xs font-mono text-slate-500">
                                  Tokens: {artifact.chunk_length_tokens}
                                </span>
                                {artifact.source && (
                                  <span className="text-xs font-mono text-slate-500 truncate">
                                    {artifact.source}
                                  </span>
                                )}
                              </div>
                              <p className="text-xs font-mono text-slate-700 leading-relaxed">
                                {artifact.text}
                              </p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  
                   
                    <div>
                      {summaryOutput && (
                        <div className="flex items-center gap-2 mb-3">
                          <div className="w-2 h-2 bg-slate-600 rounded-full"></div>
                          <h3 className="text-xs font-semibold text-slate-600 uppercase tracking-wider">
                            Final Cover Letter
                          </h3>
                        </div>
                      )}
                      <pre className="wrap-text text-sm font-mono text-slate-700 leading-7">
                        {output}
                      </pre>
                    </div>
                  </div>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-slate-300 select-none">
                    <div className="w-16 h-16 border-2 border-slate-100 rounded-full flex items-center justify-center mb-4">
                      <div className="w-2 h-2 bg-slate-200 rounded-full"></div>
                    </div>
                    <p className="text-sm">Output will appear here</p>
                  </div>
                )}
              </div>
            ) : activeTab === 'logs' ? (
              <div className="h-full overflow-auto p-6">
                {logsState.isLoading && !logsState.logs ? (
                  <div className="h-full flex flex-col items-center justify-center text-slate-400">
                    <Loader2 className="w-8 h-8 animate-spin mb-2" />
                    <p className="text-sm">Loading logs...</p>
                  </div>
                ) : logsState.logs ? (
                  <pre className={`text-xs font-mono text-slate-700 leading-6 ${logsWrapText ? 'wrap-text' : 'whitespace-pre overflow-x-auto'}`}>
                    {logsState.logs}
                  </pre>
                ) : (
                  <div className="h-full flex flex-col items-center justify-center text-slate-300 select-none">
                    <div className="w-16 h-16 border-2 border-slate-100 rounded-full flex items-center justify-center mb-4">
                      <FileText className="w-6 h-6 text-slate-200" />
                    </div>
                    <p className="text-sm">No logs available</p>
                  </div>
                )}
              </div>
            ) : (
              <div className="h-full flex flex-col relative">
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
        {showAbout && (
          <div className="absolute inset-0 z-50 bg-white/80 backdrop-blur-sm flex items-center justify-center p-4">
            <div className="bg-white w-full max-w-md border border-slate-200 shadow-xl rounded-2xl p-6">
              <h2 className="text-lg font-semibold mb-2">About This Tool</h2>
              <p className="text-slate-600 text-sm leading-relaxed mb-4">
                This is a local RAG (Retrieval-Augmented Generation) interface designed to help tailor job applications.
                It processes job descriptions locally and generates targeted documents based on your stored credentials.
              </p>
              <button
                onClick={() => setShowAbout(false)}
                className="w-full py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-lg text-sm font-medium transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        )}
      </main>
     
      <footer className="flex-none h-8 border-t border-slate-200 bg-white flex items-center justify-between px-6">
         <span className="text-[10px] text-slate-400 uppercase tracking-widest">v1.0.0 Local Build</span>
         <span className="text-[10px] text-slate-400">© {new Date().getFullYear()} Engineering</span>
      </footer>
    </div>
  );
}
export default App;