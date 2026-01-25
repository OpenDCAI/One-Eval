import { useState, useEffect, useMemo } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { 
    Clock, X, Search, Database, Play, Save, Layers, Plus, BookOpen, Trash2, AlertTriangle, Settings, ChevronRight, ChevronDown, Check
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { ChatPanel, WorkflowBlock, SummaryPanel, Bench, WorkflowState, BenchCard, GalleryModal } from "./EvalComponents";

// --- Types ---
interface StatusResponse {
  thread_id: string;
  status: "idle" | "running" | "interrupted" | "completed" | "failed" | "not_found";
  next_node: string[] | null;
  state_values: WorkflowState | null;
  current_node?: string; 
}

interface HistoryItem {
    thread_id: string;
    updated_at: string;
    user_query: string;
    status: string;
}

interface ChatMessage {
    id: string;
    role: "user" | "ai" | "system";
    content: string | React.ReactNode;
    timestamp: number;
}

export const Eval = () => {
  const [query, setQuery] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusResponse["status"]>("idle");
  const [state, setState] = useState<WorkflowState | null>(null);
  const [currentNode, setCurrentNode] = useState<string | null>(null);
  
  // History
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  
  // UI State
  const [activeNode, setActiveNode] = useState<string | null>(null);
  const [isGalleryOpen, setIsGalleryOpen] = useState(false);
  const [isResuming, setIsResuming] = useState(false); // Flag to prevent polling overwrites during resume
  const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
  const [isChatCollapsed, setIsChatCollapsed] = useState(false);
  const chatWidth = isChatCollapsed ? 60 : 400;

  // Eval Params State (Manual)
  const [evalParams, setEvalParams] = useState({
      temperature: 0.7,
      top_p: 1.0,
      max_tokens: 2048
  });

  const [expandedResults, setExpandedResults] = useState<number[]>([]);
  const [selectedModel, setSelectedModel] = useState<any | null>(null);
  
  // Chat State
  const [messages, setMessages] = useState<ChatMessage[]>([
      { id: "init", role: "ai", content: "Hello! Describe your evaluation task to get started.", timestamp: Date.now() }
  ]);

  // Editable State (for manual modification)
  const [editBenches, setEditBenches] = useState<Bench[]>([]);
  const [availableModels, setAvailableModels] = useState<any[]>([]);

  const apiBaseUrl = useMemo(() => localStorage.getItem("oneEval.apiBaseUrl") || "http://localhost:8000", []);
  
  // Fetch Models
  useEffect(() => {
      axios.get(`${apiBaseUrl}/api/models`)
          .then(res => {
              if (Array.isArray(res.data)) {
                  setAvailableModels(res.data);
              }
          })
          .catch(e => console.error("Failed to fetch models", e));
  }, [apiBaseUrl]);

  // Fetch History
  const fetchHistory = async () => {
      try {
          const res = await axios.get(`${apiBaseUrl}/api/workflow/history`);
          setHistory(Array.isArray(res.data) ? res.data : []);
      } catch (e) {
          console.error("Failed to fetch history", e);
          setHistory([]);
      }
  };

  useEffect(() => {
      fetchHistory();
  }, [apiBaseUrl, status]); 

  // Polling
  useEffect(() => {
    if (!threadId || status === "completed" || status === "failed" || isResuming) return;

    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${apiBaseUrl}/api/workflow/status/${threadId}`);
        const data: StatusResponse = res.data;
        
        // Status Transition Logic for Chat
                if (data.status !== status) {
                    // Prevent duplicate messages by checking the last message content
                    setMessages(prev => {
                        const lastMsg = prev[prev.length - 1];
                        
                        if (data.status === "failed") {
                            const failText = "Evaluation failed. Please check logs.";
                            if (lastMsg?.content !== failText) {
                                 return [...prev, { id: Date.now().toString(), role: "system", content: failText, timestamp: Date.now() }];
                            }
                        }
                        return prev;
                    });
                }

        setStatus(data.status);
        if (data.state_values) {
            setState(data.state_values);

            // Only sync editBenches if we are NOT in interrupted mode (or first time entering)
            // But if we are in interrupted mode, we want to keep user edits.
            // However, if the backend adds new benches (e.g. from search), we want them.
            // Strategy: Only sync if status changed to interrupted, or if we are not editing.
            if (data.status === "interrupted" && status !== "interrupted") {
                 setEditBenches(data.state_values.benches || []);
            }
        }
        if (data.next_node) {
            setCurrentNode(data.next_node[0]);
            setActiveNode(data.next_node[0]); // Auto-highlight active node
        }

      } catch (e) {
        console.error("Polling error", e);
      }
    }, 1500); 

    return () => clearInterval(interval);
  }, [threadId, status, isResuming]);

  // Init Edit Benches when entering interrupted state (handled in polling now for better control)
  // But also fallback here
  useEffect(() => {
      if (status === "interrupted" && state?.benches && editBenches.length === 0) {
          setEditBenches(state.benches);
      }
  }, [status, state?.benches]);

  // Sync state and params
  useEffect(() => {
      if (!state) return;

      // Sync target model
      if (state.target_model && !selectedModel) {
          setSelectedModel(state.target_model);
          
          // Sync eval params from the target model
          setEvalParams({
              temperature: state.target_model.temperature ?? 0.7,
              top_p: state.target_model.top_p ?? 1.0,
              max_tokens: state.target_model.max_tokens ?? 2048
          });
      } else if (state.target_model_name && !selectedModel && availableModels.length > 0) {
          const found = availableModels.find(m => m.name === state.target_model_name);
          if (found) setSelectedModel(found);
      }
  }, [state, availableModels.length, selectedModel]);

  const handleStart = async (userQuery: string) => {
    if (!userQuery) return;
    setQuery(userQuery);
    
    // Add User Message
    setMessages(prev => [...prev, { id: Date.now().toString(), role: "user", content: userQuery, timestamp: Date.now() }]);

    try {
      // Fetch default model
      let targetModelName = "Qwen2.5-7B";
      let targetModelPath = "/mnt/DataFlow/models/Qwen2.5-7B-Instruct";
      try {
        const modelsRes = await axios.get(`${apiBaseUrl}/api/models`);
        if (modelsRes.data && modelsRes.data.length > 0) {
            targetModelName = modelsRes.data[0].name;
            targetModelPath = modelsRes.data[0].path;
        }
      } catch (e) {}

      const res = await axios.post(`${apiBaseUrl}/api/workflow/start`, {
        user_query: userQuery,
        target_model_name: targetModelName, 
        target_model_path: targetModelPath
      });
      setThreadId(res.data.thread_id);
      setStatus("running");
      
      setMessages(prev => [...prev, { id: Date.now().toString(), role: "ai", content: "I've started the evaluation workflow. I'll analyze your query first.", timestamp: Date.now() }]);

    } catch (e) {
      console.error(e);
      setMessages(prev => [...prev, { id: Date.now().toString(), role: "system", content: "Failed to start workflow. Check connection.", timestamp: Date.now() }]);
    }
  };

  const handleResume = async () => {
    if (!threadId) return;
    
    setIsResuming(true); // Lock polling

    // Optimistic Update
    if (state) {
        setState({ ...state, benches: editBenches });
    }

    try {
      // Send updated benches if we edited them
      const payload: any = {
        thread_id: threadId,
        action: "approved",
      };
      
      if (status === "interrupted") {
          // Check if we are at Execution Confirmation step (by node or by phase logic)
          // For now, if we are interrupted and in exec phase (or prep phase finished), we attach eval params.
          // Since we don't have explicit node name for custom interrupt, we attach params generally if they exist.
          
          const nextModel = selectedModel ?? state?.target_model ?? null;
          const modelForUpdate = nextModel
              ? { ...nextModel, temperature: evalParams.temperature, top_p: evalParams.top_p, max_tokens: evalParams.max_tokens }
              : null;
          payload.state_updates = {
              benches: editBenches,
              target_model_name: selectedModel?.name ?? state?.target_model_name,
          };
          if (modelForUpdate) {
              payload.state_updates.target_model = modelForUpdate;
          }
      }

      await axios.post(`${apiBaseUrl}/api/workflow/resume/${threadId}`, payload);
      setStatus("running"); 
      setMessages(prev => [...prev, { id: Date.now().toString(), role: "ai", content: "Configuration approved. Proceeding with evaluation...", timestamp: Date.now() }]);
      
      // Re-enable polling after a delay to allow backend to process
      setTimeout(() => setIsResuming(false), 3000);

    } catch (e) {
      console.error(e);
      setIsResuming(false);
    }
  };
  
  const loadHistory = (item: HistoryItem) => {
      setThreadId(item.thread_id);
      setStatus("idle"); 
      setQuery(item.user_query);
      // Reset Chat
      setMessages([
          { id: "init", role: "ai", content: "Loaded past session.", timestamp: Date.now() },
          { id: "hist", role: "user", content: item.user_query, timestamp: Date.now() }
      ]);
      
      axios.get(`${apiBaseUrl}/api/workflow/status/${item.thread_id}`).then(res => {
          setStatus(res.data.status);
          setState(res.data.state_values);
      });
  };

  const handleNewTask = () => {
      // Disconnect from current session
      setThreadId(null);
      
      // Reset all states
      setStatus("idle");
      setQuery("");
      setState(null);
      setCurrentNode(null);
      setActiveNode(null);
      setEditBenches([]);
      
      // Reset Chat
      setMessages([
          { id: "init", role: "ai", content: "Hello! Describe your evaluation task to get started.", timestamp: Date.now() }
      ]);
  };

  const handleDeleteHistory = async (e: React.MouseEvent, threadIdToDelete: string) => {
      e.stopPropagation();
      try {
          await axios.delete(`${apiBaseUrl}/api/workflow/history/${threadIdToDelete}`);
          setHistory(prev => prev.filter(h => h.thread_id !== threadIdToDelete));
          setDeleteConfirmId(null);
          
          // If we deleted the active thread, reset
          if (threadId === threadIdToDelete) {
              handleNewTask();
          }
      } catch (err) {
          console.error("Failed to delete history item", err);
      }
  };

  // --- Bench Management ---
  const handleManualAdd = () => {
      const newBench: Bench = {
          bench_name: "new-benchmark",
          meta: {}
      };
      setEditBenches([...editBenches, newBench]);
  };

  const handleGallerySelect = (bench: any) => {
      // Check duplicate
      if (editBenches.some(b => b.bench_name === bench.bench_name)) return;
      
      const safeTaskTypes = Array.isArray(bench.task_type) 
          ? bench.task_type.map((t: any) => typeof t === 'object' ? JSON.stringify(t) : String(t))
          : [];

      const newBench: Bench = {
          bench_name: bench.bench_name,
          eval_type: safeTaskTypes.length > 0 ? safeTaskTypes[0] : "unknown",
          meta: {
              ...bench.meta,
              tags: safeTaskTypes, // Store all task types as tags
              source: "gallery", // Flag to skip probing
              skip_probing: true,
              keys: [], // Default empty keys to prevent white screen
              preview_data: [] // Default empty preview to prevent white screen
          }
      };
      setEditBenches([...editBenches, newBench]);
      setIsGalleryOpen(false);
  };

  const handleBenchUpdate = (updatedBench: Bench, index: number) => {
      const newBenches = [...editBenches];
      newBenches[index] = updatedBench;
      setEditBenches(newBenches);
      
      // Also update main state for immediate visual feedback if in interrupted mode
      if (state && status === "interrupted") {
        const newStateBenches = [...(state.benches || [])];
        if (index < newStateBenches.length) {
            newStateBenches[index] = updatedBench;
            setState({ ...state, benches: newStateBenches });
        }
      }
  };
  
  // Helper to determine block status
  const getBlockStatus = (block: 'search' | 'prep' | 'exec') => {
      if (status === 'idle') return 'idle';
      
      const nodes = currentNode ? [currentNode] : [];
      const isSearchActive = ["QueryUnderstandNode", "BenchSearchNode", "HumanReviewNode"].some(n => nodes.some(cn => cn.includes(n)));
      const isPrepActive = ["DatasetStructureNode", "BenchConfigRecommendNode", "BenchTaskInferNode", "DownloadNode"].some(n => nodes.some(cn => cn.includes(n)));
      const isExecActive = ["PreEvalReviewNode", "DataFlowEvalNode"].some(n => nodes.some(cn => cn.includes(n)));

      if (status === 'completed') return 'completed';
      
      if (block === 'search') {
          if (isSearchActive) return status === "interrupted" ? "interrupted" : "running";
          return "completed"; 
      }
      if (block === 'prep') {
          if (isPrepActive) return "running";
          if (isSearchActive) return "pending";
          return "completed";
      }
      if (block === 'exec') {
          if (isExecActive) return status === "interrupted" ? "interrupted" : "running";
          if (isSearchActive || isPrepActive) return "pending";
          return "completed";
      }
      return 'pending';
  };

  return (
    <div className="h-screen flex bg-slate-50 overflow-hidden font-['Inter']">
       {/* Background Pattern */}
       <div className="absolute inset-0 bg-[linear-gradient(to_right,#e2e8f0_1px,transparent_1px),linear-gradient(to_bottom,#e2e8f0_1px,transparent_1px)] bg-[size:2rem_2rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none opacity-50 z-0" />

       {/* --- Left Sidebar (History) --- */}
       <motion.div 
         initial={{ width: 60, opacity: 1 }}
         animate={{ width: showHistory ? 240 : 60 }}
         className="bg-white border-r border-slate-200 z-50 flex flex-col shadow-[4px_0_24px_-12px_rgba(0,0,0,0.1)] transition-all duration-300 relative"
       >
           <div className="p-4 border-b border-slate-100 flex items-center justify-between h-16 shrink-0">
               {showHistory ? (
                   <div className="flex items-center gap-2 overflow-hidden">
                       <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center text-white shadow-lg shadow-blue-500/20 shrink-0">
                           <Clock className="w-4 h-4" />
                       </div>
                       <span className="font-bold text-slate-800 truncate">History</span>
                   </div>
               ) : (
                   <div className="w-full flex justify-center">
                       <div className="w-8 h-8 rounded-lg bg-slate-100 flex items-center justify-center text-slate-500 group-hover:bg-blue-50 group-hover:text-blue-600 transition-colors cursor-pointer" onClick={() => setShowHistory(true)}>
                           <Clock className="w-5 h-5" />
                       </div>
                   </div>
               )}
               
               {showHistory && (
                   <Button variant="ghost" size="icon" onClick={() => setShowHistory(false)} className="h-8 w-8 text-slate-400 hover:text-slate-600">
                       <Layers className="w-4 h-4 rotate-90" />
                   </Button>
               )}
           </div>

           {/* New Task Button Area */}
           <div className={cn("p-3", !showHistory && "flex justify-center")}>
               <Button 
                   variant={showHistory ? "default" : "ghost"} 
                   size={showHistory ? "default" : "icon"}
                   className={cn(
                       "w-full gap-2 transition-all",
                       showHistory ? "bg-slate-900 text-white hover:bg-slate-800 shadow-md" : "h-10 w-10 rounded-xl bg-blue-50 text-blue-600 hover:bg-blue-100"
                   )}
                   onClick={handleNewTask}
                   title="Start New Task"
               >
                   <Plus className={cn("w-4 h-4", !showHistory && "w-5 h-5")} />
                   {showHistory && <span>New Task</span>}
               </Button>
           </div>
           
           <div className="flex-1 overflow-y-auto p-2 space-y-2 scrollbar-hide">
               {showHistory && Array.isArray(history) && history.map(item => {
                   if (!item || typeof item !== 'object') return null;
                   const safeThreadId = item.thread_id || `temp-${Math.random()}`;
                   const safeDate = (() => {
                       try {
                           return item.updated_at ? new Date(item.updated_at).toLocaleTimeString() : "";
                       } catch (e) {
                           return "";
                       }
                   })();

                   return (
                   <div 
                        key={safeThreadId}
                        onClick={() => loadHistory(item)}
                        className={cn(
                            "p-3 rounded-lg border cursor-pointer transition-all hover:shadow-md relative group",
                            threadId === item.thread_id ? "bg-blue-50 border-blue-200" : "bg-white border-slate-100 hover:border-slate-300"
                        )}
                   >
                       {/* Delete Overlay / Button */}
                       {deleteConfirmId === safeThreadId ? (
                           <div className="absolute inset-0 bg-white/95 z-10 flex flex-col items-center justify-center rounded-lg border border-red-100 p-2 text-center" onClick={e => e.stopPropagation()}>
                               <span className="text-[10px] text-red-600 font-bold mb-1 flex items-center gap-1">
                                   <AlertTriangle className="w-3 h-3" /> Confirm Delete?
                               </span>
                               <div className="flex gap-2 w-full">
                                   <Button 
                                       size="sm" 
                                       variant="outline" 
                                       className="h-6 flex-1 text-[10px] p-0" 
                                       onClick={(e) => {
                                           e.stopPropagation();
                                           setDeleteConfirmId(null);
                                       }}
                                   >
                                       Cancel
                                   </Button>
                                   <Button 
                                       size="sm" 
                                       className="h-6 flex-1 text-[10px] p-0 bg-red-500 hover:bg-red-600 text-white" 
                                       onClick={(e) => handleDeleteHistory(e, safeThreadId)}
                                   >
                                       Delete
                                   </Button>
                               </div>
                           </div>
                       ) : (
                           <Button
                               variant="ghost"
                               size="icon"
                               className="absolute top-2 right-2 h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity text-slate-400 hover:text-red-500 hover:bg-red-50 z-10"
                               onClick={(e) => {
                                   e.stopPropagation();
                                   setDeleteConfirmId(safeThreadId);
                               }}
                               title="Delete Task"
                           >
                               <Trash2 className="w-3 h-3" />
                           </Button>
                       )}

                       <div className="flex justify-between items-start mb-1">
                           <span className={cn(
                               "text-[10px] uppercase font-bold px-1.5 py-0.5 rounded",
                               item.status === "completed" ? "bg-green-100 text-green-700" :
                               item.status === "interrupted" ? "bg-amber-100 text-amber-700" :
                               "bg-slate-100 text-slate-600"
                           )}>{item.status || "UNKNOWN"}</span>
                           <span className="text-[10px] text-slate-400">{safeDate}</span>
                       </div>
                       <p className="text-xs text-slate-700 font-medium line-clamp-2 pr-4" title={item.user_query}>
                           {item.user_query || "Untitled Task"}
                       </p>
                   </div>
                   );
               })}
           </div>
       </motion.div>

       {/* --- Center Canvas --- */}
       <div className="flex-1 flex flex-col relative z-10 h-full overflow-hidden">
           
           {/* Top Toolbar */}
           <header className="px-6 h-16 flex justify-between items-center bg-white/80 backdrop-blur-md border-b border-slate-200 z-20">
             <div className="flex items-center gap-2">
                <h2 className="font-bold text-lg text-slate-900 tracking-tight flex items-center gap-2">
                    OneEval <span className="text-blue-600">Studio</span>
                </h2>
             </div>
             
             <div className="flex items-center gap-3">
                 <Button variant="outline" size="sm" className="gap-2">
                     <Database className="w-4 h-4" /> Benches
                 </Button>
                 <Button variant="outline" size="sm" className="gap-2">
                     <Layers className="w-4 h-4" /> Task
                 </Button>
                 <Button variant="outline" size="sm" className="gap-2">
                     <Save className="w-4 h-4" /> Save
                 </Button>
                 <div className="w-px h-6 bg-slate-200 mx-1" />
                 <Button 
                    size="sm" 
                    className={cn(
                        "gap-2 transition-all",
                        status === "running" ? "bg-red-500 hover:bg-red-600" : "bg-blue-600 hover:bg-blue-700"
                    )}
                 >
                     {status === "running" ? <><X className="w-4 h-4" /> Stop</> : <><Play className="w-4 h-4" /> Run</>}
                 </Button>
             </div>
           </header>

           {/* Blocks Canvas */}
           <main className="flex-1 overflow-y-auto p-8 pb-32 scroll-smooth">
               <div className="max-w-5xl mx-auto space-y-12">
                   
                   {/* Block 1: Discovery */}
                   <WorkflowBlock 
                        title="Discovery Phase" 
                        icon={Search}
                        activeNodeId={activeNode}
                        status={getBlockStatus('search') as any}
                        colorTheme="violet"
                        nodes={[
                            { id: "QueryUnderstandNode", label: "Understand" },
                            { id: "BenchSearchNode", label: "Search" },
                            { id: "HumanReviewNode", label: "Review" }
                        ]}
                   >
                       <div className="space-y-8 relative">
                           {/* Node 1: Understand */}
                           <div className="pl-6 border-l-2 border-violet-100 relative">
                               <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-violet-50 border-2 border-violet-200" />
                               <h4 className="text-xs font-bold text-violet-600 uppercase tracking-wider mb-3">1. Understand</h4>
                               
                               <div className="space-y-3">
                                   <div>
                                       <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider pl-1">User Query</label>
                                       <div className="p-3 bg-slate-50/50 rounded-lg border border-slate-100 text-sm text-slate-700 shadow-inner">
                                           {query || <span className="text-slate-400 italic">Waiting for input...</span>}
                                       </div>
                                   </div>
                                   {/* Domain (Placeholder/Mock if not in state) */}
                                   {(state as any)?.domain && (
                                       <div>
                                           <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider pl-1">Identified Domain</label>
                                           <div className="flex flex-wrap gap-2 mt-1">
                                               {Array.isArray((state as any).domain) 
                                                   ? (state as any).domain.map((d: string) => <span key={d} className="px-2 py-1 bg-violet-100 text-violet-700 text-xs rounded-md font-bold">{d}</span>)
                                                   : <span className="px-2 py-1 bg-violet-100 text-violet-700 text-xs rounded-md font-bold">{(state as any).domain}</span>
                                               }
                                           </div>
                                       </div>
                                   )}
                               </div>
                           </div>

                           {/* Node 2: Search */}
                           <div className="pl-6 border-l-2 border-violet-100 relative">
                               <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-violet-50 border-2 border-violet-200" />
                               <h4 className="text-xs font-bold text-violet-600 uppercase tracking-wider mb-3">2. Search</h4>
                               
                               {/* Editable Benches List */}
                               {state?.benches?.length ? (
                                   <div className="space-y-3">
                                       <div className="flex justify-between items-center px-1">
                                           <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">Target Benchmarks</label>
                                           {status === "interrupted" && (
                                               <div className="flex gap-2">
                                                   <Button size="sm" variant="outline" className="h-6 px-2 text-[10px] gap-1 bg-amber-50 border-amber-200 text-amber-700 hover:bg-amber-100" onClick={handleManualAdd}>
                                                        <Plus className="w-3 h-3" /> Add Custom
                                                   </Button>
                                                   <Button size="sm" variant="outline" className="h-6 px-2 text-[10px] gap-1 bg-blue-50 border-blue-200 text-blue-700 hover:bg-blue-100" onClick={() => setIsGalleryOpen(true)}>
                                                        <BookOpen className="w-3 h-3" /> From Gallery
                                                   </Button>
                                               </div>
                                           )}
                                       </div>
                                       
                                       <div className="grid grid-cols-1 gap-3">
                                           {(status === "interrupted" ? editBenches : state.benches).map((b, i) => (
                                               <div key={i} className={cn(
                                                   "flex items-center gap-4 p-3 rounded-xl border transition-all",
                                                   status === "interrupted" 
                                                       ? "bg-white border-amber-200 shadow-sm shadow-amber-100" 
                                                       : "bg-slate-50/50 border-slate-100"
                                               )}>
                                                   <div className="w-8 h-8 rounded-lg bg-violet-100 text-violet-600 flex items-center justify-center text-xs font-bold shrink-0">
                                                       {b.bench_name.substring(0, 2).toUpperCase()}
                                                   </div>
                                                   {status === "interrupted" ? (
                                                       <div className="flex flex-1 flex-col gap-1">
                                                           <div className="flex items-center gap-2">
                                                              <Input 
                                                                  placeholder="Enter benchmark name..."
                                                                  value={b.bench_name}
                                                                  onChange={(e) => {
                                                                        const nb = [...editBenches];
                                                                        nb[i].bench_name = e.target.value;
                                                                        setEditBenches(nb);
                                                                   }}
                                                                   className="h-9 text-sm border-amber-100 focus-visible:ring-amber-500 bg-amber-50/30"
                                                               />
                                                               <Button 
                                                                   variant="ghost" 
                                                                   size="icon" 
                                                                   className="h-9 w-9 text-amber-600 hover:bg-amber-100 hover:text-amber-700"
                                                                   onClick={() => {
                                                                       const nb = editBenches.filter((_, idx) => idx !== i);
                                                                       setEditBenches(nb);
                                                                   }}
                                                               >
                                                                   <X className="w-4 h-4" />
                                                               </Button>
                                                           </div>
                                                           {b.meta?.desc && (
                                                               <span className="text-xs text-slate-400 pl-1 truncate" title={typeof b.meta.desc === 'string' ? b.meta.desc : (b.meta.desc ? JSON.stringify(b.meta.desc) : '')}>
                                                                   {typeof b.meta.desc === 'string' ? b.meta.desc : (b.meta.desc ? JSON.stringify(b.meta.desc) : '')}
                                                               </span>
                                                           )}
                                                       </div>
                                                   ) : (
                                                       <div className="flex flex-col">
                                                           {/* Updated to show desc */}
                                                           <span className="font-mono font-medium text-sm text-slate-700">{b.bench_name}</span>
                                                           {b.meta?.desc && (
                                                               <span className="text-xs text-slate-400 max-w-md truncate" title={typeof b.meta.desc === 'string' ? b.meta.desc : (b.meta.desc ? JSON.stringify(b.meta.desc) : '')}>
                                                                   {typeof b.meta.desc === 'string' ? b.meta.desc : (b.meta.desc ? JSON.stringify(b.meta.desc) : '')}
                                                               </span>
                                                           )}
                                                       </div>
                                                   )}
                                               </div>
                                           ))}
                                       </div>
                                   </div>
                               ) : (
                                   <div className="text-sm text-slate-400 italic pl-1">No benchmarks selected yet.</div>
                               )}
                           </div>

                           {/* Node 3: Review */}
                           <div className="pl-6 border-l-2 border-transparent relative">
                               <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-violet-50 border-2 border-violet-200" />
                               <h4 className="text-xs font-bold text-violet-600 uppercase tracking-wider mb-3">3. Review</h4>
                               
                               <div className="space-y-2">
                                   <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider pl-1">User Notes</label>
                                   <textarea 
                                       className="w-full p-3 bg-slate-50/50 rounded-lg border border-slate-100 text-sm text-slate-700 shadow-inner resize-none focus:outline-none focus:ring-1 focus:ring-violet-200"
                                       rows={2}
                                       placeholder="Add any specific instructions or notes for this evaluation..."
                                   />
                               </div>
                           </div>
                       </div>
                   </WorkflowBlock>

                   {/* Block 2: Preparation */}
                   <WorkflowBlock 
                        title="Preparation Phase" 
                        icon={Database}
                        activeNodeId={activeNode}
                        status={getBlockStatus('prep') as any}
                        colorTheme="amber"
                        nodes={[
                            { id: "DatasetStructureNode", label: "Structure" },
                            { id: "BenchConfigRecommendNode", label: "Config" },
                            { id: "BenchTaskInferNode", label: "Inference" },
                            { id: "DownloadNode", label: "Download" }
                        ]}
                   >
                       {/* Config View */}
                       <div className="grid grid-cols-2 gap-4">
                           {/* Use editBenches if interrupted to show live updates, else state.benches */}
                           {(status === "interrupted" ? editBenches : state?.benches)?.map((b, i) => (
                               <div key={i} className="h-48">
                                   <BenchCard 
                                       bench={b} 
                                       activeNode={activeNode} 
                                       onUpdate={(updated) => handleBenchUpdate(updated, i)}
                                   />
                               </div>
                           ))}
                           {!(state?.benches?.length || editBenches.length) && (
                               <div className="col-span-2 py-8 flex flex-col items-center justify-center text-slate-300 border-2 border-dashed border-slate-100 rounded-xl">
                                   <Database className="w-8 h-8 mb-2 opacity-50" />
                                   <span className="text-sm">No benchmarks configured</span>
                               </div>
                           )}
                       </div>
                   </WorkflowBlock>

                   {/* Block 3: Execution */}
                   <WorkflowBlock 
                        title="Execution Phase" 
                        icon={Play}
                        activeNodeId={activeNode}
                        status={getBlockStatus('exec') as any}
                        colorTheme="emerald"
                        nodes={[
                            { id: "PreEvalReviewNode", label: "Confirm" },
                            { id: "DataFlowEvalNode", label: "Evaluation" }
                        ]}
                   >
                       <div className="space-y-6">
                           {/* Eval Config Section */}
                           <div className={cn(
                               "bg-emerald-50/50 p-4 rounded-xl border space-y-4 transition-all",
                               status === "interrupted" && currentNode?.includes("PreEvalReviewNode")
                                   ? "border-amber-400 ring-2 ring-amber-100 shadow-lg shadow-amber-50" 
                                   : "border-emerald-100"
                           )}>
                               <div className="flex justify-between items-center">
                                   <div className="flex items-center gap-2 text-emerald-800 font-bold text-sm">
                                       <Settings className="w-4 h-4" /> Evaluation Configuration
                                   </div>
                                   
                                   {/* Status Indicator */}
                                   {status === "interrupted" && currentNode?.includes("PreEvalReviewNode") && (
                                       <span className="text-[10px] font-bold text-amber-600 bg-amber-100 px-2 py-1 rounded animate-pulse">
                                           Waiting for Confirmation
                                       </span>
                                   )}
                               </div>
                               <div className="grid grid-cols-12 gap-x-4 gap-y-4">
                                   <div className="col-span-12 min-w-0">
                                       <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Target Model</label>
                                       {availableModels && availableModels.length > 0 ? (
                                           <select
                                               value={(selectedModel?.name ?? state?.target_model_name ?? "") as any}
                                               onChange={(e) => {
                                                   const found = availableModels.find((m: any) => m?.name === e.target.value);
                                                   if (found) setSelectedModel(found);
                                               }}
                                               disabled={status === "running"}
                                               className="w-full h-9 rounded-lg border border-emerald-200 bg-white px-3 text-sm font-bold text-slate-900 shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 transition-all disabled:opacity-50 disabled:bg-slate-50/50"
                                           >
                                               {availableModels.map((m: any) => (
                                                   <option key={m.name} value={m.name}>{m.name}</option>
                                               ))}
                                           </select>
                                       ) : (
                                           <div className="h-9 flex items-center px-3 rounded-lg border border-emerald-200 bg-slate-50/50 text-sm font-bold text-slate-500 italic">
                                               {state?.target_model_name || "No model selected"}
                                           </div>
                                       )}
                                   </div>
                                   
                                   <div className="col-span-4 min-w-0">
                                       <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Temperature</label>
                                       <Input 
                                           type="number" 
                                           step="0.1"
                                           min="0"
                                           max="2"
                                           value={evalParams.temperature} 
                                           onChange={e => setEvalParams({...evalParams, temperature: parseFloat(e.target.value) || 0})}
                                           disabled={status === "running"}
                                           className="h-9 bg-white border-emerald-200 rounded-lg focus-visible:ring-emerald-500/20 focus-visible:border-emerald-500 text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                       />
                                   </div>
                                   
                                   <div className="col-span-4 min-w-0">
                                       <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Top P</label>
                                       <Input 
                                           type="number" 
                                           step="0.05"
                                           min="0"
                                           max="1"
                                           value={evalParams.top_p} 
                                           onChange={e => setEvalParams({...evalParams, top_p: parseFloat(e.target.value) || 0})}
                                           disabled={status === "running"}
                                           className="h-9 bg-white border-emerald-200 rounded-lg focus-visible:ring-emerald-500/20 focus-visible:border-emerald-500 text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                       />
                                   </div>
                                   
                                   <div className="col-span-4 min-w-0">
                                       <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Max Tokens</label>
                                       <Input 
                                           type="number" 
                                           step="128"
                                           value={evalParams.max_tokens} 
                                           onChange={e => setEvalParams({...evalParams, max_tokens: parseInt(e.target.value) || 0})}
                                           disabled={status === "running"}
                                           className="h-9 bg-white border-emerald-200 rounded-lg focus-visible:ring-emerald-500/20 focus-visible:border-emerald-500 text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                       />
                                   </div>
                               </div>
                           </div>

                           <div className="space-y-3">
                               {state?.benches?.map((b, i) => {
                                   const isExpanded = expandedResults.includes(i);
                                   const res = b.meta?.eval_result;
                                       const pickScore = (r: any) => {
                                           if (!r || typeof r !== 'object') return null;
                                           for (const k of ['score','accuracy','exact_match']) {
                                               const v = (r as any)[k];
                                               if (typeof v === 'number') return v;
                                               if (typeof v === 'string') return v;
                                           }
                                           for (const v of Object.values(r)) {
                                               if (typeof v === 'number' || typeof v === 'string') return v;
                                           }
                                           return null;
                                       };
                                       const score = pickScore(res);
                                   
                                   return (
                                       <div key={i} className="bg-white rounded-xl border border-slate-100 shadow-sm overflow-hidden transition-all hover:border-emerald-200">
                                           <div 
                                               className="flex items-center justify-between p-4 cursor-pointer hover:bg-slate-50/50"
                                               onClick={() => {
                                                   if (expandedResults.includes(i)) setExpandedResults(expandedResults.filter(idx => idx !== i));
                                                   else setExpandedResults([...expandedResults, i]);
                                               }}
                                           >
                                               <div className="flex items-center gap-3">
                                                   <div className={cn(
                                                       "w-2 h-8 rounded-full transition-colors",
                                                       b.eval_status === "success" ? "bg-emerald-500" : "bg-slate-200"
                                                   )} />
                                                   <div>
                                                       <div className="text-sm font-bold text-slate-700">{b.bench_name}</div>
                                                       <div className="text-[10px] text-slate-400 flex items-center gap-2">
                                                           {b.download_status === "success" && <span className="flex items-center gap-1"><Database className="w-3 h-3" /> Ready</span>}
                                                           {b.eval_status === "success" && <span className="flex items-center gap-1 text-emerald-600"><Check className="w-3 h-3" /> Evaluated</span>}
                                                       </div>
                                                   </div>
                                               </div>
                                               
                                               <div className="flex items-center gap-4">
                                                   {score !== null ? (
                                                       <div className="flex items-center gap-3 bg-emerald-50 px-3 py-1.5 rounded-lg border border-emerald-100">
                                                           <span className="text-xs font-bold text-emerald-600 uppercase tracking-wider">Score</span>
                                                           <span className="text-lg font-black text-emerald-700 font-mono">
                                                                {typeof score === 'number' ? score.toFixed(2) : String(score)}
                                                           </span>
                                                       </div>
                                                   ) : (
                                                       <span className="text-xs text-slate-400 italic">Waiting for results...</span>
                                                   )}
                                                   <Button variant="ghost" size="icon" className="h-6 w-6 text-slate-400">
                                                       {isExpanded ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
                                                   </Button>
                                               </div>
                                           </div>
                                           
                                           {/* Expanded Details */}
                                           {isExpanded && (
                                               <div className="px-4 pb-4 pt-0 bg-slate-50/30 border-t border-slate-50">
                                                   <div className="grid grid-cols-2 gap-4 mt-4">
                                                       <div className="p-3 bg-white rounded-lg border border-slate-100">
                                                           <div className="text-[10px] text-slate-400 uppercase font-bold mb-1">Total Samples</div>
                                                           <div className="text-lg font-mono font-bold text-slate-700">
                                                               {b.meta?.download_config?.count || b.meta?.structure?.count || "N/A"}
                                                           </div>
                                                       </div>
                                                       <div className="p-3 bg-white rounded-lg border border-slate-100">
                                                           <div className="text-[10px] text-slate-400 uppercase font-bold mb-1">Metrics Detail</div>
                                                           <div className="text-xs font-mono text-slate-600 space-y-1">
                                                               {res ? Object.entries(res).map(([key, value]) => (
                                                                   <div key={key} className="flex justify-between border-b border-slate-50 pb-1 last:border-0">
                                                                       <span className="text-slate-400 mr-2">{key}:</span>
                                                                       <span className="font-bold text-slate-700 truncate" title={String(value)}>{String(value)}</span>
                                                                   </div>
                                                               )) : "No detailed metrics"}
                                                           </div>
                                                       </div>
                                                   </div>
                                               </div>
                                           )}
                                       </div>
                                   );
                               })}
                               
                               {!state?.benches?.length && (
                                   <div className="py-8 flex flex-col items-center justify-center text-slate-300 border-2 border-dashed border-slate-100 rounded-xl">
                                       <Play className="w-8 h-8 mb-2 opacity-50" />
                                       <span className="text-sm">Ready to execute</span>
                                   </div>
                               )}
                               
                               {/* Summary Footer */}
                               {state?.benches?.length && state.benches.some(b => b.eval_status === "success") && (
                                   <div className="mt-6 p-4 bg-slate-800 text-white rounded-xl shadow-lg flex justify-between items-center">
                                       <div className="flex gap-6">
                                           <div>
                                               <div className="text-[10px] text-slate-400 uppercase font-bold">Benchmarks</div>
                                               <div className="text-xl font-bold">{state.benches.length}</div>
                                           </div>
                                           <div>
                                               <div className="text-[10px] text-slate-400 uppercase font-bold">Total Samples</div>
                                               <div className="text-xl font-bold">
                                                   {state.benches.reduce((acc, b) => acc + (parseInt(b.meta?.download_config?.count || b.meta?.structure?.count || 0)), 0)}
                                               </div>
                                           </div>
                                       </div>
                                       <div className="text-right">
                                           <div className="text-[10px] text-emerald-400 uppercase font-bold">Overall Status</div>
                                           <div className="text-sm font-bold text-emerald-100">Evaluation Completed</div>
                                       </div>
                                   </div>
                               )}
                           </div>
                       </div>
                   </WorkflowBlock>

               </div>
           </main>
           
           {/* Bottom Summary Panel */}
           <SummaryPanel 
                state={state} 
                sidebarWidth={showHistory ? 240 : 60} 
                chatWidth={chatWidth}
           />
       </div>

       {/* --- Right Sidebar (Chat) --- */}
       <div className="h-full z-40 shadow-2xl relative flex-shrink-0">
           <ChatPanel 
                messages={messages} 
                status={status}
                onSendMessage={handleStart}
                onConfirm={handleResume}
                isWaitingForInput={status !== "idle"}
                activeNodeId={activeNode} 
                isCollapsed={isChatCollapsed}
                onToggleCollapse={() => setIsChatCollapsed(!isChatCollapsed)}
           />
       </div>

       {/* Gallery Modal */}
       <GalleryModal 
            isOpen={isGalleryOpen} 
            onClose={() => setIsGalleryOpen(false)} 
            onSelect={handleGallerySelect}
            apiBaseUrl={apiBaseUrl}
       />

    </div>
  );
};

export default Eval;
