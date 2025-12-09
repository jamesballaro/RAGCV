import { useState, useEffect } from 'react';
import { Send, Copy, Check, Info, Loader2, FileText, RefreshCw } from 'lucide-react';

// LaTeX escaping utility
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
    '^': '\\^{}',
    '~': '\\~{}'
  };
  
  return text.replace(/[\\{}$&%#_^~]/g, (match) => replacements[match] || match);
};

// LaTeX preamble template
const LATEX_PREAMBLE = `\\documentclass[12pt]{article}
\\usepackage[a4paper,margin=0.8in,top=0.5in]{geometry}
\\usepackage{hyperref}
\\usepackage{enumitem}
\\usepackage{changepage} % Needed for the adjustwidth environment
\\hyphenpenalty = 10000
\\usepackage{parskip} % for spacing between paragraphs
\\usepackage[T1]{fontenc}
\\usepackage{lmodern}
\\usepackage{sectsty}
\\usepackage{setspace}
\\setstretch{1.14}
\\emergencystretch 10pt
% \\usepackage[none]{hyphenat}
\\sectionfont{\\fontsize{13}{0}\\selectfont}
\\raggedbottom
\\begin{document}
\\pagestyle{empty}
`;

const LATEX_POSTAMBLE = `\n\\end{document}`;

function App() {
  const [input, setInput] = useState('');
  const [output, setOutput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState(null);
  const [showAbout, setShowAbout] = useState(false);
  const [activeTab, setActiveTab] = useState('output');
  const [logs, setLogs] = useState('');
  const [logsLoading, setLogsLoading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(false);
  
  // LaTeX-specific state
  const [latexSource, setLatexSource] = useState('');
  const [pdfUrl, setPdfUrl] = useState('');
  const [latexLoading, setLatexLoading] = useState(false);
  const [latexError, setLatexError] = useState(null);

  // Fetch logs from server
  const fetchLogs = async () => {
    setLogsLoading(true);
    try {
      const response = await fetch('http://localhost:8000/logs');
      if (!response.ok) throw new Error(`Failed to fetch logs: ${response.status}`);
      const data = await response.json();
      setLogs(data.logs || 'No logs available');
    } catch (err) {
      setLogs(`Error loading logs: ${err.message}`);
    } finally {
      setLogsLoading(false);
    }
  };

  // Compile LaTeX to PDF
  const compileLaTeX = async (plainTextOutput) => {
    setLatexLoading(true);
    setLatexError(null);
    
    // Clean up old PDF URL
    if (pdfUrl) {
      URL.revokeObjectURL(pdfUrl);
      setPdfUrl('');
    }

    try {
      // Escape the plain text and wrap in preamble
      const escapedContent = escapeLaTeX(plainTextOutput);
      const fullLatex = LATEX_PREAMBLE + escapedContent + LATEX_POSTAMBLE;
      setLatexSource(fullLatex);

      // Send to backend for compilation
      const response = await fetch('http://localhost:8000/compile_latex', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ latex: fullLatex }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'LaTeX compilation failed');
      }

      // Create blob URL from PDF response
      const pdfBlob = await response.blob();
      const url = URL.createObjectURL(pdfBlob);
      setPdfUrl(url);
    } catch (err) {
      setLatexError(err.message || "Failed to compile LaTeX");
    } finally {
      setLatexLoading(false);
    }
  };

  // Auto-refresh logs when enabled
  useEffect(() => {
    if (autoRefresh && activeTab === 'logs') {
      const interval = setInterval(fetchLogs, 2000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, activeTab]);

  // Fetch logs when switching to logs tab
  useEffect(() => {
    if (activeTab === 'logs' && !logs) {
      fetchLogs();
    }
  }, [activeTab]);

  // Cleanup PDF URL on unmount
  useEffect(() => {
    return () => {
      if (pdfUrl) {
        URL.revokeObjectURL(pdfUrl);
      }
    };
  }, [pdfUrl]);

  const handleSubmit = async () => {
    if (!input.trim()) return;

    setIsLoading(true);
    setError(null);
    setOutput('');
    setLatexError(null);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: input }),
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const data = await response.json();
      const resultText = data.result || JSON.stringify(data, null, 2);
      setOutput(resultText);
      
      // Auto-compile LaTeX in background
      compileLaTeX(resultText);
      
      // Auto-fetch logs after query completes
      if (activeTab === 'logs') {
        fetchLogs();
      }
    } catch (err) {
      setError(err.message || "Failed to connect to the assistant.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleCopy = () => {
    let textToCopy = '';
    if (activeTab === 'output') textToCopy = output;
    else if (activeTab === 'logs') textToCopy = logs;
    else if (activeTab === 'latex') textToCopy = latexSource;
    
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-white text-slate-900 font-sans selection:bg-indigo-50 selection:text-indigo-900">
      
      {/* Header */}
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

      {/* Main Workspace */}
      <main className="flex-grow flex flex-col lg:flex-row overflow-hidden relative">
        
        {/* LEFT PANE: Input */}
        <div className="flex-1 flex flex-col border-r border-slate-200 h-1/2 lg:h-full relative group">
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

          {/* Floating Action Button */}
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

        {/* RIGHT PANE: Output/Logs/LaTeX with Tabs */}
        <div className="flex-1 flex flex-col h-1/2 lg:h-full bg-slate-50/30">
          {/* Tab Header */}
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
                        onChange={(e) => setAutoRefresh(e.target.checked)}
                        className="w-3 h-3 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                      />
                      Auto-refresh
                    </label>
                    <button
                      onClick={fetchLogs}
                      disabled={logsLoading}
                      className="flex items-center gap-1 text-xs font-medium text-slate-500 hover:text-indigo-600 transition-colors disabled:opacity-50"
                    >
                      <RefreshCw className={`w-3.5 h-3.5 ${logsLoading ? 'animate-spin' : ''}`} />
                      Refresh
                    </button>
                  </>
                )}
                {((activeTab === 'output' && output) || 
                  (activeTab === 'logs' && logs) || 
                  (activeTab === 'latex' && latexSource)) && (
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

          {/* Content Area */}
          <div className="flex-grow overflow-auto relative">
            {activeTab === 'output' ? (
              // Output Tab
              <div className="h-full overflow-auto p-6">
                {error ? (
                  <div className="h-full flex flex-col items-center justify-center text-red-500">
                    <p className="font-mono text-sm">Error: {error}</p>
                  </div>
                ) : output ? (
                  <pre className="wrap-text text-sm font-mono text-slate-700 leading-7">
                    {output}
                  </pre>
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
              // Logs Tab
              <div className="h-full overflow-auto p-6">
                {logsLoading && !logs ? (
                  <div className="h-full flex flex-col items-center justify-center text-slate-400">
                    <Loader2 className="w-8 h-8 animate-spin mb-2" />
                    <p className="text-sm">Loading logs...</p>
                  </div>
                ) : logs ? (
                  <pre className="wrap-text text-xs font-mono text-slate-700 leading-6">
                    {logs}
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
              // LaTeX PDF Tab
              <div className="h-full flex flex-col">
                {latexLoading ? (
                  <div className="h-full flex flex-col items-center justify-center text-slate-400">
                    <Loader2 className="w-8 h-8 animate-spin mb-2" />
                    <p className="text-sm">Compiling LaTeX...</p>
                  </div>
                ) : latexError ? (
                  <div className="h-full flex flex-col items-center justify-center text-red-500 p-6">
                    <p className="font-mono text-sm mb-2">LaTeX Compilation Error:</p>
                    <p className="font-mono text-xs text-center">{latexError}</p>
                  </div>
                ) : pdfUrl ? (
                  <>
                    {/* PDF Preview - Top Half */}
                    <div className="flex-1 border-b border-slate-200 bg-slate-100">
                      <iframe
                        src={pdfUrl}
                        className="w-full h-full"
                        title="LaTeX PDF Preview"
                      />
                    </div>
                    
                    {/* LaTeX Source - Bottom Half */}
                    <div className="flex-1 flex flex-col bg-white">
                      <div className="flex-none px-6 py-2 bg-slate-50 border-b border-slate-100">
                        <span className="text-xs font-mono font-medium text-slate-400 uppercase tracking-wider">
                          LaTeX Source (Read-Only)
                        </span>
                      </div>
                      <textarea
                        className="flex-grow w-full p-6 resize-none outline-none text-xs font-mono leading-relaxed text-slate-700 bg-white"
                        value={latexSource}
                        readOnly
                        spellCheck="false"
                      />
                    </div>
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

        {/* About Modal */}
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
      
      {/* Footer */}
      <footer className="flex-none h-8 border-t border-slate-200 bg-white flex items-center justify-between px-6">
         <span className="text-[10px] text-slate-400 uppercase tracking-widest">v1.0.0 Local Build</span>
         <span className="text-[10px] text-slate-400">© {new Date().getFullYear()} Engineering</span>
      </footer>
    </div>
  );
}

export default App;