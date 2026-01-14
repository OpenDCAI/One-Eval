import { useState, useEffect, useMemo } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Settings, Loader2, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

// --- Types ---
interface Bench {
  bench_name: string;
  eval_type?: string;
  meta?: any;
  eval_status?: string;
}

interface WorkflowState {
  user_query: string;
  benches: Bench[];
  target_model: string;
}

interface StatusResponse {
  status: "idle" | "running" | "interrupted" | "completed" | "failed";
  next_node: string[] | null;
  state_values: WorkflowState | null;
}

// --- Node Component (Horizontal) ---
const NodeStep = ({ 
    label, 
    status, 
    isLast 
}: { 
    label: string; 
    status: "pending" | "running" | "completed" | "error";
    isLast?: boolean;
}) => {
    return (
        <div className="flex items-center">
            {/* Node Circle */}
            <div className="relative flex flex-col items-center group">
                <div className={cn(
                    "w-8 h-8 rounded-full flex items-center justify-center border-2 transition-all duration-300 z-10",
                    status === "completed" ? "bg-slate-900 border-slate-900 text-white" :
                    status === "running" ? "bg-white border-blue-500 text-blue-500 shadow-[0_0_0_4px_rgba(59,130,246,0.1)]" :
                    "bg-white border-slate-200 text-slate-300"
                )}>
                    {status === "completed" ? <Check className="w-4 h-4" /> : 
                     status === "running" ? <div className="w-2 h-2 bg-current rounded-full animate-ping" /> :
                     <div className="w-2 h-2 bg-current rounded-full" />
                    }
                </div>
                <span className={cn(
                    "absolute top-10 text-xs font-medium whitespace-nowrap transition-colors",
                    status === "pending" ? "text-slate-400" : "text-slate-700"
                )}>
                    {label}
                </span>
            </div>

            {/* Connector Line */}
            {!isLast && (
                <div className={cn(
                    "h-[2px] w-12 mx-2 transition-colors duration-500",
                    status === "completed" ? "bg-slate-900" : "bg-slate-100"
                )} />
            )}
        </div>
    );
};


// --- Workflow Card Component ---
const WorkflowCard = ({ 
  title, 
  isActive, 
  status,
  children,
}: { 
  title: string; 
  isActive: boolean; 
  status: "pending" | "running" | "completed";
  children: React.ReactNode;
}) => {
  return (
    <motion.div
      layout
      className={cn(
        "bg-white rounded-2xl border transition-all duration-500 overflow-hidden flex flex-col",
        isActive 
          ? "border-slate-300 shadow-xl shadow-slate-200/50 min-h-[300px]" 
          : "border-slate-100 shadow-sm opacity-60 min-h-[100px]"
      )}
    >
      <div className="px-6 py-4 border-b border-slate-50 bg-slate-50/50 flex justify-between items-center">
        <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
            <div className={cn(
                "w-2 h-2 rounded-full",
                status === "completed" ? "bg-green-500" :
                status === "running" ? "bg-blue-500 animate-pulse" :
                "bg-slate-300"
            )} />
            {title}
        </h3>
        {isActive && <Settings className="w-4 h-4 text-slate-400 cursor-pointer hover:text-slate-700" />}
      </div>
      
      <div className="p-6 flex-1 flex flex-col justify-center">
        {children}
      </div>
    </motion.div>
  );
};

export const Eval = () => {
  const [query, setQuery] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusResponse["status"]>("idle");
  const [state, setState] = useState<WorkflowState | null>(null);
  const [currentNode, setCurrentNode] = useState<string | null>(null);

  const apiBaseUrl = useMemo(() => localStorage.getItem("oneEval.apiBaseUrl") || "http://localhost:8000", []);
  
  // Polling
  useEffect(() => {
    if (!threadId || status === "completed" || status === "failed") return;

    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${apiBaseUrl}/api/workflow/status/${threadId}`);
        const data: StatusResponse = res.data;
        
        setStatus(data.status);
        if (data.state_values) setState(data.state_values);
        if (data.next_node) setCurrentNode(data.next_node[0]);

      } catch (e) {
        console.error("Polling error", e);
      }
    }, 2000); 

    return () => clearInterval(interval);
  }, [threadId, status]);

  const handleStart = async () => {
    if (!query) return;
    try {
      const res = await axios.post(`${apiBaseUrl}/api/workflow/start`, {
        user_query: query,
        target_model_name: "Qwen2.5-7B", 
        target_model_path: "/mnt/DataFlow/models/Qwen2.5-7B-Instruct" 
      });
      setThreadId(res.data.thread_id);
      setStatus("running");
      setQuery(""); 
    } catch (e) {
      console.error(e);
    }
  };

  const handleResume = async () => {
    if (!threadId) return;
    try {
      await axios.post(`${apiBaseUrl}/api/workflow/resume/${threadId}`, {
        action: "approved",
        selected_benches: [] 
      });
      setStatus("running"); 
    } catch (e) {
      console.error(e);
    }
  };

  // Node Status Helper
  const getNodeStatus = (nodeName: string) => {
    if (currentNode === nodeName) return "running";
    const order = [
        "QueryUnderstandNode", "BenchSearchNode", "HumanReviewNode",
        "DatasetStructureNode", "BenchConfigRecommendNode", "DownloadNode",
        "DatasetKeysNode", "BenchTaskInferNode", "DataFlowEvalNode"
    ];
    const currIdx = order.indexOf(currentNode || "");
    const nodeIdx = order.indexOf(nodeName);
    if (currIdx > nodeIdx) return "completed";
    return "pending";
  };

  const isSearchActive = ["QueryUnderstandNode", "BenchSearchNode", "HumanReviewNode"].some(n => currentNode?.includes(n));
  const isPrepActive = ["DatasetStructureNode", "BenchConfigRecommendNode", "DownloadNode"].some(n => currentNode?.includes(n));
  const isExecActive = ["DatasetKeysNode", "BenchTaskInferNode", "DataFlowEvalNode"].some(n => currentNode?.includes(n));
  
  const getBlockStatus = (block: 'search' | 'prep' | 'exec') => {
      if (status === 'idle') return 'pending';
      if (status === 'completed') return 'completed';
      
      if (block === 'search') return isSearchActive ? 'running' : 'completed';
      if (block === 'prep') return isPrepActive ? 'running' : (isSearchActive ? 'pending' : 'completed');
      if (block === 'exec') return isExecActive ? 'running' : (isSearchActive || isPrepActive ? 'pending' : 'completed');
      return 'pending';
  }

  return (
    <div className="h-screen flex flex-col bg-slate-50 relative overflow-hidden font-['Inter']">
       {/* Background */}
       <div className="absolute inset-0 bg-[linear-gradient(to_right,#e2e8f0_1px,transparent_1px),linear-gradient(to_bottom,#e2e8f0_1px,transparent_1px)] bg-[size:2rem_2rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none opacity-50" />
       
       {/* Top Bar */}
       <header className="px-8 py-4 flex justify-between items-center z-20 border-b border-slate-200 bg-white/50 backdrop-blur-sm">
         <div className="flex items-center gap-4">
            <h2 className="font-bold text-lg text-slate-900 tracking-tight">Evaluation Workflow</h2>
            {threadId && <span className="text-xs font-mono text-slate-400 bg-slate-100 px-2 py-1 rounded">ID: {threadId.split('-')[0]}</span>}
         </div>
         <div className="flex items-center gap-3">
             {status === "running" && <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />}
             <span className={cn(
                 "text-xs font-medium uppercase tracking-wider px-2 py-1 rounded-full",
                 status === "running" ? "bg-blue-50 text-blue-700" :
                 status === "completed" ? "bg-green-50 text-green-700" :
                 "bg-slate-100 text-slate-500"
             )}>{status}</span>
         </div>
       </header>

       {/* Main Layout */}
       <main className="flex-1 px-8 py-6 flex flex-col gap-6 overflow-y-auto z-10 max-w-5xl mx-auto w-full pb-32">
         
         {/* Block 1: Discovery */}
         <WorkflowCard 
            title="Discovery Phase" 
            status={getBlockStatus('search') as any}
            isActive={status === 'idle' || isSearchActive || status === 'interrupted'}
         >
             <div className="flex flex-col gap-8">
                {/* Node Flow */}
                <div className="flex items-center justify-center py-4">
                    <NodeStep label="Understand" status={getNodeStatus("QueryUnderstandNode")} />
                    <NodeStep label="Search" status={getNodeStatus("BenchSearchNode")} />
                    <NodeStep label="Review" status={getNodeStatus("HumanReviewNode")} isLast />
                </div>

                {/* Content */}
                <AnimatePresence mode="wait">
                    {state?.benches?.length ? (
                        <motion.div 
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            className="bg-slate-50 border border-slate-100 rounded-xl p-4 grid grid-cols-1 md:grid-cols-2 gap-3"
                        >
                            {state.benches.map((b, i) => (
                                <div key={i} className="flex items-center justify-between p-3 bg-white rounded-lg shadow-sm border border-slate-100">
                                    <span className="font-medium text-sm text-slate-700">{b.bench_name}</span>
                                    {status === "interrupted" && <div className="w-4 h-4 rounded-full border-2 border-blue-500 bg-blue-500" />}
                                </div>
                            ))}
                        </motion.div>
                    ) : (
                        <div className="text-center text-slate-400 text-sm py-4 italic">
                            Waiting for workflow to start...
                        </div>
                    )}
                </AnimatePresence>

                {status === "interrupted" && (
                    <div className="flex justify-center">
                        <Button
                          onClick={handleResume}
                          className="text-white bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 shadow-lg shadow-blue-600/20"
                        >
                            Approve Benchmarks & Continue
                        </Button>
                    </div>
                )}
             </div>
         </WorkflowCard>

         {/* Block 2: Planning */}
         <WorkflowCard 
            title="Preparation Phase" 
            status={getBlockStatus('prep') as any}
            isActive={isPrepActive}
         >
             <div className="flex flex-col gap-8">
                 <div className="flex items-center justify-center py-4">
                    <NodeStep label="Analyze" status={getNodeStatus("DatasetStructureNode")} />
                    <NodeStep label="Config" status={getNodeStatus("BenchConfigRecommendNode")} />
                    <NodeStep label="Download" status={getNodeStatus("DownloadNode")} isLast />
                </div>
                
                {state?.benches?.length && isPrepActive && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {state.benches.map((b, i) => (
                             <div key={i} className="bg-slate-50 p-4 rounded-lg border border-slate-100">
                                 <div className="text-xs font-bold text-slate-500 mb-2 uppercase tracking-wider">{b.bench_name}</div>
                                 {b.meta?.key_mapping ? (
                                     <div className="text-xs font-mono text-slate-600 bg-white p-2 rounded border border-slate-100">
                                         {JSON.stringify(b.meta.key_mapping, null, 2)}
                                     </div>
                                 ) : (
                                     <div className="flex items-center gap-2 text-xs text-blue-600">
                                        <Loader2 className="w-3 h-3 animate-spin" /> Processing schema...
                                     </div>
                                 )}
                             </div>
                        ))}
                    </div>
                )}
            </div>
         </WorkflowCard>

         {/* Block 3: Execution */}
         <WorkflowCard 
            title="Execution Phase" 
            status={getBlockStatus('exec') as any}
            isActive={isExecActive || status === 'completed'}
         >
             <div className="flex flex-col gap-8">
                 <div className="flex items-center justify-center py-4">
                    <NodeStep label="Infer Task" status={getNodeStatus("BenchTaskInferNode")} />
                    <NodeStep label="Evaluate" status={getNodeStatus("DataFlowEvalNode")} isLast />
                </div>

                <div className="space-y-3">
                     {state?.benches?.map((b, i) => (
                        <div key={i} className="bg-white p-4 rounded-xl border border-slate-100 shadow-sm flex flex-col gap-4">
                             <div className="flex justify-between items-center">
                                 <span className="font-semibold text-slate-900">{b.bench_name}</span>
                                 {b.eval_status === "success" && (
                                     <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
                                         Completed
                                     </span>
                                 )}
                             </div>
                             
                             {b.meta?.eval_result && (
                                <div className="flex gap-4">
                                    {Object.entries(b.meta.eval_result).map(([k, v]) => (
                                        <div key={k} className="flex flex-col">
                                            <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">{k}</span>
                                            <span className="text-lg font-mono font-medium text-slate-900">{String(v)}</span>
                                        </div>
                                    ))}
                                </div>
                             )}
                        </div>
                     ))}
                </div>
            </div>
         </WorkflowCard>

       </main>

       {/* Bottom Chat Area */}
       <div className="fixed bottom-0 left-0 right-0 p-6 z-30 flex justify-center pointer-events-none">
         <motion.div 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            className="w-full max-w-2xl pointer-events-auto"
         >
            <div className="bg-white border border-slate-200 rounded-2xl flex items-center p-2 shadow-2xl shadow-slate-200/50">
                <Input 
                    placeholder={status === "idle" ? "Describe your evaluation task..." : "Workflow is running..."}
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    disabled={status !== "idle"}
                    className="border-0 bg-transparent focus-visible:ring-0 text-slate-900 placeholder:text-slate-400 h-12 text-lg shadow-none"
                    onKeyDown={e => e.key === "Enter" && status === "idle" && handleStart()}
                />
                <Button 
                    size="icon" 
                    onClick={handleStart} 
                    disabled={status !== "idle" || !query}
                    className={cn(
                        "h-10 w-10 rounded-xl transition-all shadow-sm", 
                        query
                          ? "text-white bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 shadow-blue-600/20"
                          : "bg-slate-100 text-slate-400"
                    )}
                >
                    {status === "idle" ? <Send className="w-5 h-5" /> : <div className="w-2 h-2 bg-current rounded-full animate-ping" />}
                </Button>
            </div>
         </motion.div>
       </div>
    </div>
  );
};
