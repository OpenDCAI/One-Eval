import React, { useRef, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
    Send, Check, Loader2, AlertCircle, ChevronDown, ChevronUp, 
    Database, Bot, Maximize2, X, Save as SaveIcon, Tag
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";

// --- Types ---
export interface Bench {
  bench_name: string;
  eval_type?: string;
  meta?: any;
  eval_status?: string;
  download_status?: string;
}

export interface WorkflowState {
  user_query: string;
  benches: Bench[];
  target_model_name?: string;
  target_model?: any;
}

// --- Modal Component ---
const Modal = ({ isOpen, onClose, title, description, children, footer }: { isOpen: boolean, onClose: () => void, title: React.ReactNode, description?: string, children: React.ReactNode, footer?: React.ReactNode }) => {
    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-[100] flex items-center justify-center p-4">
            <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 bg-black/40 backdrop-blur-sm"
                onClick={onClose}
            />
            <motion.div 
                initial={{ opacity: 0, scale: 0.95, y: 10 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.95, y: 10 }}
                className="bg-white rounded-2xl shadow-2xl w-full max-w-3xl max-h-[85vh] flex flex-col relative z-10 overflow-hidden"
            >
                <div className="p-6 border-b border-slate-100 flex items-start justify-between bg-slate-50/50">
                    <div>
                        <h3 className="text-xl font-bold text-slate-900 flex items-center gap-2">{title}</h3>
                        {description && <p className="text-sm text-slate-500 mt-1">{description}</p>}
                    </div>
                    <Button variant="ghost" size="icon" onClick={onClose} className="h-8 w-8 -mr-2 -mt-2">
                        <X className="w-5 h-5 text-slate-400" />
                    </Button>
                </div>
                <div className="flex-1 overflow-y-auto p-6">
                    {children}
                </div>
                {footer && (
                    <div className="p-4 border-t border-slate-100 bg-slate-50/50 flex justify-end gap-2">
                        {footer}
                    </div>
                )}
            </motion.div>
        </div>
    );
};

// --- Bench Card Component ---
export const BenchCard = ({ bench, activeNode, onUpdate }: { bench: Bench, activeNode: string | null, onUpdate?: (updatedBench: Bench) => void }) => {
    const [isDetailsOpen, setIsDetailsOpen] = useState(false);
    
    // Local state for editing in modal
    const [editKeyMap, setEditKeyMap] = useState<Record<string, string>>({});
    const [selectedSubset, setSelectedSubset] = useState<string>("");
    const [selectedSplit, setSelectedSplit] = useState<string>("");
    
    // Safe parsing helper
    const safeParse = (data: any) => {
        if (!data) return null;
        if (typeof data === 'string') {
            try { return JSON.parse(data); } catch { return data; }
        }
        return data;
    };

    // Helper to check if value is a plain object (not array, not null)
    const isObject = (val: any) => val != null && typeof val === 'object' && !Array.isArray(val);
    
    const toLabel = (val: any): string => {
        if (val == null) return "";
        if (typeof val === "string") return val;
        if (typeof val === "number" || typeof val === "boolean") return String(val);
        if (typeof val === "object") {
            const name = (val as any).name;
            if (typeof name === "string") return name;
            try { return JSON.stringify(val); } catch { return String(val); }
        }
        return String(val);
    };

    // Extract Meta Data safely
    const meta = bench.meta || {};
    const structure = safeParse(meta.structure); 
    const keyMapping = safeParse(meta.key_mapping); 
    const downloadConfig = safeParse(meta.download_config); 
    const evalType = bench.eval_type || meta.bench_dataflow_eval_type; 
    const descriptionRaw = meta.card_text ?? meta.description ?? meta.desc ?? "No description available.";
    const description = typeof descriptionRaw === 'string' ? descriptionRaw : (descriptionRaw ? JSON.stringify(descriptionRaw) : "No description available.");
    const tags = Array.isArray(meta.tags) ? meta.tags : [];
    const availableKeys = Array.isArray(meta.keys) ? meta.keys : []; // Extracted keys from dataset
    const previewData = Array.isArray(meta.preview_data) ? meta.preview_data : []; // Preview rows
    const downloadPath = meta.download_path || meta.local_path;
    const sampleCountRaw = downloadConfig?.count || structure?.count || meta.count;
    const sampleCount = (() => {
        if (typeof sampleCountRaw === 'number') return sampleCountRaw;
        if (typeof sampleCountRaw === 'string') return sampleCountRaw;
        if (sampleCountRaw && typeof sampleCountRaw === 'object') {
            const num = (sampleCountRaw as any).num_examples ?? (sampleCountRaw as any).count;
            if (typeof num === 'number' || typeof num === 'string') return num;
            return JSON.stringify(sampleCountRaw);
        }
        return undefined;
    })();

    // Parse structure for selector
    const structureSubsetsRaw = Array.isArray(structure?.subsets) ? structure.subsets : [];
    const structureSubsets = structureSubsetsRaw
        .map((s: any) => {
            const subsetVal = s?.subset ?? s?.name ?? s;
            const subset = toLabel(subsetVal);
            const splitsRaw = s?.splits ?? [];
            const splits = Array.isArray(splitsRaw) ? splitsRaw.map(toLabel).filter(Boolean) : [];
            return { subset, splits };
        })
        .filter((s: any) => s?.subset);

    // Init state from bench
    useEffect(() => {
        if (keyMapping) {
            setEditKeyMap({ ...keyMapping });
        }
        if (downloadConfig) {
            setSelectedSubset(toLabel(downloadConfig.config) || "default");
            setSelectedSplit(toLabel(downloadConfig.split) || "test");
        } else if (structureSubsets.length > 0) {
            // Default selection if no config exists
            const first = structureSubsets[0];
            setSelectedSubset(toLabel(first.subset));
            if (first.splits && first.splits.length > 0) {
                setSelectedSplit(toLabel(first.splits[0]));
            }
        }
    }, [bench, isDetailsOpen]);

    const handleSave = () => {
        if (!onUpdate) return;
        
        const updatedBench = { ...bench };
        if (!updatedBench.meta) updatedBench.meta = {};
        
        // Update Key Map
        updatedBench.meta.key_mapping = editKeyMap;
        
        // Update Config
        const currentReason = updatedBench.meta.download_config?.reason || "User manually selected configuration.";
        
        updatedBench.meta.download_config = {
             config: selectedSubset,
             split: selectedSplit,
             reason: currentReason
        };

        onUpdate(updatedBench);
        setIsDetailsOpen(false);
    };

    // Determine status color
    const statusColor = bench.download_status === "success" ? "bg-green-50 text-green-600" : "bg-slate-50 text-slate-400";
    
    // Check if we have meaningful data to show
    const hasData = structure || keyMapping || downloadConfig || evalType || (meta && Object.keys(meta).length > 0);
    
    // Determine loading state based on active node
    const isLoading = !hasData && (
        activeNode === "DatasetStructureNode" || 
        activeNode === "BenchConfigRecommendNode" || 
        activeNode === "BenchTaskInferNode"
    );

    return (
        <>
            <div 
                onClick={() => hasData && setIsDetailsOpen(true)}
                className={cn(
                    "p-4 bg-white rounded-xl border border-slate-100 shadow-sm transition-all relative h-full flex flex-col group",
                    hasData ? "hover:shadow-md cursor-pointer hover:border-blue-200" : "cursor-default"
                )}
            >
                <div className="flex justify-between mb-2 shrink-0">
                    <span className="text-sm font-bold text-slate-700 truncate pr-2" title={bench.bench_name}>{bench.bench_name}</span>
                    <span className={cn(
                        "text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider h-fit whitespace-nowrap",
                        statusColor
                    )}>
                        {bench.download_status || "Pending"}
                    </span>
                </div>
                
                {/* Type & Tags */}
                <div className="flex flex-wrap gap-1 mb-3">
                    {evalType && (
                        <span className="text-[10px] bg-blue-50 text-blue-600 px-1.5 py-0.5 rounded border border-blue-100 font-medium">
                            {typeof evalType === 'object' ? JSON.stringify(evalType) : String(evalType)}
                        </span>
                    )}
                    {tags.slice(0, 3).map((tag: any, i: number) => {
                         const tagStr = typeof tag === 'object' ? JSON.stringify(tag) : String(tag);
                         return (
                            <span key={i} className="text-[10px] bg-slate-50 text-slate-500 px-1.5 py-0.5 rounded border border-slate-100 flex items-center gap-1">
                                <Tag className="w-2 h-2" /> {tagStr}
                            </span>
                         );
                    })}
                </div>

                <div className="text-[10px] text-slate-500 font-mono flex-1 overflow-hidden bg-slate-50/50 p-3 rounded-lg border border-slate-50 group-hover:border-slate-100 transition-colors relative">
                    {hasData ? (
                        <div className="space-y-3 h-full overflow-y-auto pb-4 scrollbar-hide">
                            <div className="text-slate-500 mb-2 font-sans leading-relaxed text-[10px] whitespace-pre-wrap line-clamp-4">
                                {description}
                            </div>

                            {/* Key Mapping Preview */}
                            {keyMapping && isObject(keyMapping) && (
                                <div>
                                    <div className="text-[9px] uppercase font-bold text-slate-400 mb-1">Key Mapping</div>
                                    <div className="grid grid-cols-1 gap-1 pl-2 border-l-2 border-amber-200">
                                        {Object.entries(keyMapping).slice(0, 3).map(([k, v]) => (
                                            <div key={k} className="flex gap-1">
                                                <span className="text-slate-600">{k}:</span>
                                                <span className="text-amber-700 font-bold truncate" title={String(v)}>{String(v)}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Download Info */}
                            {(downloadPath || sampleCount) && (
                                <div className="pt-2 border-t border-slate-200/50 flex justify-between items-center text-[9px] text-slate-400 font-mono">
                                    {sampleCount !== undefined && <span className="flex items-center gap-1"><Database className="w-2 h-2" /> {String(sampleCount)}</span>}
                                    {downloadPath && <span className="truncate max-w-[100px] flex items-center gap-1" title={downloadPath}><SaveIcon className="w-2 h-2" /> ...{downloadPath.slice(-12)}</span>}
                                </div>
                            )}
                        </div>
                    ) : (
                        <div className="h-full flex flex-col items-center justify-center text-slate-400 gap-2 min-h-[100px]">
                            {isLoading ? (
                                <>
                                    <Loader2 className="w-4 h-4 animate-spin text-amber-500" />
                                    <span className="text-xs italic">Analyzing...</span>
                                </>
                            ) : (
                                <span className="text-xs italic">Waiting for analysis...</span>
                            )}
                        </div>
                    )}
                    
                    {/* Hover Hint */}
                    {hasData && (
                        <div className="absolute bottom-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity bg-white shadow-sm border border-slate-200 rounded px-2 py-1 text-[10px] text-blue-600 font-bold pointer-events-none flex items-center gap-1">
                            <Maximize2 className="w-3 h-3" /> Configure
                        </div>
                    )}
                </div>
            </div>

            <AnimatePresence>
                {isDetailsOpen && (
                    <Modal
                        isOpen={isDetailsOpen}
                        onClose={() => setIsDetailsOpen(false)}
                        title={
                            <>
                                <Database className="w-6 h-6 text-blue-600" />
                                {bench.bench_name}
                                {evalType && (
                                    <span className="text-sm font-normal text-slate-400 ml-2 bg-slate-100 px-2 py-0.5 rounded-full">
                                        {typeof evalType === 'object' ? JSON.stringify(evalType) : String(evalType)}
                                    </span>
                                )}
                            </>
                        }
                        description="Review and modify the benchmark configuration."
                        footer={
                            <>
                                <Button variant="ghost" onClick={() => setIsDetailsOpen(false)}>Cancel</Button>
                                <Button onClick={handleSave} className="bg-blue-600 hover:bg-blue-700 text-white gap-2">
                                    <SaveIcon className="w-4 h-4" /> Save Configuration
                                </Button>
                            </>
                        }
                    >
                        <div className="space-y-8">
                            
                            {/* 1. Key Mapping Section (Editable) */}
                            {Object.keys(editKeyMap).length > 0 && (
                                <div className="bg-amber-50/50 p-5 rounded-xl border border-amber-100">
                                    <div className="flex justify-between items-center mb-4">
                                        <h4 className="text-sm font-bold text-amber-800 flex items-center gap-2">
                                            <span className="w-2 h-2 rounded-full bg-amber-500"/> Key Mapping
                                        </h4>
                                        {evalType && (
                                            <div className="text-xs text-amber-700 bg-amber-100 px-2 py-1 rounded border border-amber-200">
                                                Type: <b>{evalType}</b>
                                            </div>
                                        )}
                                    </div>
                                    <div className="grid grid-cols-2 gap-x-8 gap-y-4 text-sm">
                                        {Object.entries(editKeyMap).map(([k, v]) => (
                                            <div key={k} className="flex items-center gap-3">
                                                <span className="text-slate-500 font-mono w-1/3 text-right text-xs truncate" title={k}>{k}</span>
                                                {availableKeys.length > 0 ? (
                                                    <select 
                                                        value={String(v)}
                                                        onChange={(e) => setEditKeyMap({...editKeyMap, [k]: e.target.value})}
                                                        className="h-8 bg-white border border-amber-200 rounded px-2 text-xs font-mono text-slate-800 focus:outline-none focus:ring-2 focus:ring-amber-500 w-full"
                                                    >
                                                        <option value="">Select key...</option>
                                                        {availableKeys.map((ak: any) => (
                                                            <option key={String(ak)} value={String(ak)}>{String(ak)}</option>
                                                        ))}
                                                    </select>
                                                ) : (
                                                    <Input 
                                                        value={String(v)}
                                                        onChange={(e) => setEditKeyMap({...editKeyMap, [k]: e.target.value})}
                                                        className="h-8 bg-white border-amber-200 focus-visible:ring-amber-500 font-mono text-xs font-bold text-slate-800"
                                                    />
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}

                            {/* Dataset Preview Section */}
                            {previewData.length > 0 && (
                                <div className="bg-slate-50 p-5 rounded-xl border border-slate-100">
                                    <h4 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-blue-500"/> Dataset Preview
                                    </h4>
                                    <div className="overflow-x-auto rounded-lg border border-slate-200">
                                        <table className="w-full text-xs text-left">
                                            <thead className="text-[10px] text-slate-500 uppercase bg-slate-100 font-bold">
                                                <tr>
                                                    {Object.keys(previewData[0] || {}).map(h => <th key={h} className="px-3 py-2 whitespace-nowrap">{h}</th>)}
                                                </tr>
                                            </thead>
                                            <tbody className="divide-y divide-slate-100">
                                                {previewData.slice(0, 5).map((row: any, idx: number) => (
                                                    <tr key={idx} className="bg-white hover:bg-slate-50">
                                                        {isObject(row) ? Object.values(row).map((val: any, vi) => (
                                                            <td key={vi} className="px-3 py-2 font-mono text-slate-600 max-w-[200px] truncate border-r border-slate-50 last:border-r-0" title={String(val)}>
                                                                {String(val)}
                                                            </td>
                                                        )) : <td className="px-3 py-2 text-slate-400 italic">Invalid Row Data</td>}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                        <div className="bg-slate-50 px-3 py-1 text-[10px] text-slate-400 text-center border-t border-slate-200">
                                            Showing first {Math.min(previewData.length, 5)} rows
                                        </div>
                                    </div>
                                </div>
                            )}

                            {/* 2. Config Selector Section */}
                            {structureSubsets.length > 0 && (
                                <div className="bg-slate-50 p-5 rounded-xl border border-slate-100">
                                    <h4 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-slate-500"/> Dataset Configuration
                                    </h4>
                                    
                                    <div className="space-y-4">
                                        {/* Subsets */}
                                        <div>
                                            <label className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block">Subset / Config</label>
                                            <div className="flex flex-wrap gap-2">
                                                {structureSubsets.map((s: any) => (
                                                    <button
                                                        key={s.subset}
                                                        onClick={() => {
                                                            setSelectedSubset(s.subset);
                                                            // Auto-select first split if available
                                                            if (s.splits && s.splits.length > 0) setSelectedSplit(s.splits[0]);
                                                        }}
                                                        className={cn(
                                                            "px-3 py-1.5 rounded-lg text-xs font-medium border transition-all",
                                                            selectedSubset === s.subset 
                                                                ? "bg-slate-800 text-white border-slate-800 shadow-md" 
                                                                : "bg-white text-slate-600 border-slate-200 hover:border-slate-300"
                                                        )}
                                                    >
                                                        {toLabel(s.subset)}
                                                    </button>
                                                ))}
                                            </div>
                                        </div>

                                        {/* Splits (Based on selected subset) */}
                                        {selectedSubset && (
                                            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                                                <label className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block">Split</label>
                                                <div className="flex flex-wrap gap-2">
                                                    {structureSubsets.find((s: any) => s.subset === selectedSubset)?.splits?.map((split: any) => (
                                                        <button
                                                            key={toLabel(split)}
                                                            onClick={() => setSelectedSplit(toLabel(split))}
                                                            className={cn(
                                                                "px-3 py-1.5 rounded-lg text-xs font-medium border transition-all",
                                                                selectedSplit === toLabel(split)
                                                                    ? "bg-blue-600 text-white border-blue-600 shadow-md" 
                                                                    : "bg-white text-slate-600 border-slate-200 hover:border-slate-300"
                                                            )}
                                                        >
                                                            {toLabel(split)}
                                                        </button>
                                                    ))}
                                                </div>
                                            </motion.div>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* 3. Result Preview / Agent Reason */}
                            <div className="bg-blue-50/50 p-5 rounded-xl border border-blue-100">
                                <div className="flex justify-between items-center mb-3">
                                    <h4 className="text-sm font-bold text-blue-800 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-blue-500"/> Final Configuration
                                    </h4>
                                    <span className="text-[10px] text-blue-400 uppercase font-bold tracking-wider">
                                        Agent Recommended Configuration
                                    </span>
                                </div>
                                <div className="bg-white p-3 rounded-lg border border-blue-100 flex gap-4 items-center">
                                    <div className="flex-1">
                                        <div className="text-[10px] text-slate-400 uppercase">Config</div>
                                        <div className="font-mono font-bold text-blue-700">{selectedSubset || "Not selected"}</div>
                                    </div>
                                    <div className="w-px h-8 bg-blue-100" />
                                    <div className="flex-1">
                                        <div className="text-[10px] text-slate-400 uppercase">Split</div>
                                        <div className="font-mono font-bold text-blue-700">{selectedSplit || "Not selected"}</div>
                                    </div>
                                </div>
                                {downloadConfig?.reason && (
                                    <div className="mt-2 text-xs text-blue-600/80 italic pl-1">
                                        "{downloadConfig.reason}"
                                    </div>
                                )}
                            </div>
                            
                            {/* Raw Meta Fallback */}
                            <div className="border-t border-slate-100 pt-4">
                                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Raw Metadata</h4>
                                <pre className="text-[10px] text-slate-400 max-h-32 overflow-y-auto bg-slate-50 p-2 rounded border border-slate-100">
                                    {JSON.stringify(bench.meta, null, 2)}
                                </pre>
                            </div>
                        </div>
                    </Modal>
                )}
            </AnimatePresence>
        </>
    );
};

// --- Chat Panel Component ---
interface ChatMessage {
    id: string;
    role: "user" | "ai" | "system";
    content: string | React.ReactNode;
    timestamp: number;
}

interface ChatPanelProps {
    messages: ChatMessage[];
    status: string;
    onSendMessage: (msg: string) => void;
    onConfirm: () => void;
    isWaitingForInput: boolean;
    activeNodeId?: string | null;
    isCollapsed: boolean;
    onToggleCollapse: () => void;
}

const EMOJIS = ["✨", "🤖", "🚀", "💡", "🔮", "✅", "🎯"];

export const ChatPanel = ({ messages, status, onSendMessage, onConfirm, isWaitingForInput, activeNodeId, isCollapsed, onToggleCollapse }: ChatPanelProps) => {
    const [input, setInput] = React.useState("");
    const [hasApproved, setHasApproved] = React.useState(false);
    const confirmText = activeNodeId?.includes("PreEvalReviewNode")
        ? "Please review the target model and evaluation parameters in the Execution Phase, then approve to start evaluation."
        : "I've prepared the benchmark configuration. Please review the highlighted parameters in the workflow blocks.";
    
    // Reset approved state
    useEffect(() => {
        if (status === "interrupted") {
             setHasApproved(false);
        }
    }, [activeNodeId]);

    const handleConfirm = () => {
        setHasApproved(true);
        onConfirm();
    };

    const scrollRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages]);

    const handleSend = () => {
        if (!input.trim()) return;
        onSendMessage(input);
        setInput("");
    };
    
    const getRandomEmoji = () => EMOJIS[Math.floor(Math.random() * EMOJIS.length)];

    return (
        <motion.div 
            animate={{ width: isCollapsed ? 60 : 400 }}
            className="h-full flex flex-col bg-white/60 backdrop-blur-xl border-l border-white/40 shadow-[-10px_0_30px_-10px_rgba(0,0,0,0.1)] relative overflow-hidden transition-all duration-300"
        >
            {/* Collapse Toggle */}
            <Button 
                variant="ghost" 
                size="icon" 
                onClick={onToggleCollapse}
                className="absolute top-4 right-4 z-50 h-6 w-6 text-slate-400 hover:text-slate-600"
            >
                {isCollapsed ? <ChevronDown className="w-4 h-4 rotate-90" /> : <ChevronDown className="w-4 h-4 -rotate-90" />}
            </Button>

            {isCollapsed ? (
                <div className="flex flex-col items-center pt-20 gap-4">
                    <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-white shadow-lg">
                        <Bot className="w-5 h-5" />
                    </div>
                    <div className={cn("w-2 h-2 rounded-full", status === "running" ? "bg-green-500 animate-pulse" : "bg-slate-300")} />
                </div>
            ) : (
                <>
                    {/* Header */}
                    <div className="p-4 border-b border-white/40 bg-white/30 flex items-center justify-between pr-12">
                        <div className="flex items-center gap-2 text-slate-800 font-bold">
                            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-500 to-fuchsia-500 flex items-center justify-center text-white shadow-lg shadow-violet-500/20">
                                <Bot className="w-5 h-5" />
                            </div>
                            <span>OneEval Assistant</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <div className={cn("w-2 h-2 rounded-full", status === "running" ? "bg-green-500 animate-pulse" : "bg-slate-300")} />
                            <span className="text-xs text-slate-500 uppercase font-medium">{status}</span>
                        </div>
                    </div>

                    {/* Messages */}
                    <div className="flex-1 overflow-y-auto p-4 space-y-6" ref={scrollRef}>
                        {messages.map((msg) => (
                            <motion.div 
                                key={msg.id}
                                initial={{ opacity: 0, y: 10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className={cn(
                                    "flex flex-col max-w-[90%]",
                                    msg.role === "user" ? "self-end items-end" : "self-start items-start"
                                )}
                            >
                                <div className="flex items-center gap-2 mb-1 px-1">
                                    {msg.role === "ai" && <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">AI Assistant</span>}
                                    {msg.role === "user" && <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">You</span>}
                                </div>
                                
                                <div className={cn(
                                    "px-5 py-3.5 text-sm shadow-sm leading-relaxed",
                                    msg.role === "user" 
                                        ? "bg-gradient-to-br from-blue-600 to-indigo-600 text-white rounded-2xl rounded-tr-sm shadow-blue-500/20" 
                                        : "bg-white border border-white/60 text-slate-700 rounded-2xl rounded-tl-sm shadow-sm"
                                )}>
                                    {msg.role === "ai" && <span className="mr-2">{getRandomEmoji()}</span>}
                                    {msg.content}
                                </div>
                                <span className="text-[10px] text-slate-300 mt-1 px-1">
                                    {new Date(msg.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                                </span>
                            </motion.div>
                        ))}

                        {/* Confirmation Card */}
                        {status === "interrupted" && !hasApproved && (
                            <motion.div 
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="self-start w-full pr-8"
                            >
                                <div className="bg-gradient-to-br from-white to-amber-50/50 border border-amber-100 rounded-2xl p-5 shadow-lg shadow-amber-500/5 ring-1 ring-amber-100/50 relative overflow-hidden">
                                    <div className="absolute top-0 right-0 w-16 h-16 bg-amber-500/10 rounded-full blur-2xl -mr-8 -mt-8" />
                                    
                                    <div className="flex items-center gap-2 mb-3 text-amber-600 font-bold text-sm">
                                        <AlertCircle className="w-4 h-4" />
                                        Review Required
                                    </div>
                                    <p className="text-sm text-slate-600 mb-5 leading-relaxed">
                                        {confirmText}
                                    </p>
                                    <div className="flex gap-3">
                                        <Button size="sm" onClick={handleConfirm} className="flex-1 bg-amber-500 hover:bg-amber-600 text-white shadow-lg shadow-amber-500/20 border-0 rounded-xl h-9">
                                            <Check className="w-4 h-4 mr-1.5" /> Approve
                                        </Button>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                        
                        {status === "running" && (
                             <div className="self-start flex items-center gap-2 text-xs text-slate-400 pl-2 bg-slate-50/50 px-3 py-1.5 rounded-full border border-slate-100">
                                 <Loader2 className="w-3 h-3 animate-spin text-blue-500" /> 
                                 <span>Processing workflow...</span>
                             </div>
                        )}
                    </div>

                    {/* Input Area */}
                    <div className="p-4 bg-white/40 border-t border-white/20 backdrop-blur-md">
                        <div className="relative group">
                            <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-500 to-violet-500 rounded-2xl opacity-20 group-hover:opacity-40 transition duration-500 blur"></div>
                            <div className="relative flex items-center bg-white rounded-xl shadow-sm border border-slate-100 overflow-hidden">
                                <Input 
                                    value={input}
                                    onChange={e => setInput(e.target.value)}
                                    onKeyDown={e => e.key === "Enter" && handleSend()}
                                    placeholder="Type a message..."
                                    disabled={isWaitingForInput}
                                    className="border-0 bg-transparent focus-visible:ring-0 text-slate-800 placeholder:text-slate-400 h-12 text-sm px-4 shadow-none"
                                />
                                <Button 
                                    size="icon" 
                                    variant="ghost" 
                                    className="mr-1 h-9 w-9 text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                                    onClick={handleSend}
                                    disabled={!input.trim() || isWaitingForInput}
                                >
                                    <Send className="w-4 h-4" />
                                </Button>
                            </div>
                        </div>
                        <div className="text-[10px] text-center text-slate-400 mt-2">
                            Press Enter to send
                        </div>
                    </div>
                </>
            )}
        </motion.div>
    );
};

// --- Workflow Block Component ---
interface WorkflowBlockProps {
    title: string;
    icon: React.ElementType;
    nodes: { id: string; label: string }[];
    activeNodeId: string | null;
    status: "pending" | "running" | "completed" | "interrupted" | "idle";
    colorTheme: "violet" | "amber" | "emerald";
    children: React.ReactNode;
}

export const WorkflowBlock = ({ title, icon: Icon, nodes, activeNodeId, status, colorTheme, children }: WorkflowBlockProps) => {
    // Theme configurations
    const themes = {
        violet: {
            bg: "bg-violet-50",
            border: "border-violet-100",
            text: "text-violet-700",
            iconBg: "bg-violet-100",
            iconText: "text-violet-600",
            activeBorder: "border-violet-300",
            shadow: "shadow-violet-500/10",
            gradient: "from-violet-50 to-white",
            nodeActive: "bg-violet-500 border-violet-500 text-white"
        },
        amber: {
            bg: "bg-amber-50",
            border: "border-amber-100",
            text: "text-amber-700",
            iconBg: "bg-amber-100",
            iconText: "text-amber-600",
            activeBorder: "border-amber-300",
            shadow: "shadow-amber-500/10",
            gradient: "from-amber-50 to-white",
            nodeActive: "bg-amber-500 border-amber-500 text-white"
        },
        emerald: {
            bg: "bg-emerald-50",
            border: "border-emerald-100",
            text: "text-emerald-700",
            iconBg: "bg-emerald-100",
            iconText: "text-emerald-600",
            activeBorder: "border-emerald-300",
            shadow: "shadow-emerald-500/10",
            gradient: "from-emerald-50 to-white",
            nodeActive: "bg-emerald-500 border-emerald-500 text-white"
        }
    };
    
    const theme = themes[colorTheme];
    const isBlockActive = nodes.some(n => n.id === activeNodeId) || status === "completed" || status === "interrupted"; 
    
    // Auto-expand if active or completed, but also if it contains meaningful content (hacky check: if status is idle, collapse)
    // Actually just use isBlockActive logic which is fine.

    return (
        <motion.div 
            layout
            className={cn(
                "rounded-[2rem] border transition-all duration-500 overflow-hidden flex flex-col relative group",
                isBlockActive 
                    ? `bg-white border-transparent shadow-xl ${theme.shadow} ring-1 ring-slate-100` 
                    : "bg-white/60 border-slate-100 shadow-sm opacity-80 hover:opacity-100"
            )}
        >
            {/* Header / Node Visualization */}
            <div className={cn("p-6 border-b border-slate-50 bg-gradient-to-b", theme.gradient)}>
                <div className="flex justify-between items-start mb-8">
                    <div className="flex items-center gap-4">
                        <div className={cn(
                            "w-12 h-12 rounded-2xl flex items-center justify-center shadow-sm",
                            theme.iconBg, theme.iconText
                        )}>
                            <Icon className="w-6 h-6" />
                        </div>
                        <div>
                            <h3 className={cn("font-bold text-xl tracking-tight", isBlockActive ? "text-slate-900" : "text-slate-500")}>{title}</h3>
                            <div className="flex items-center gap-2 mt-1">
                                <span className={cn("w-2 h-2 rounded-full", isBlockActive ? theme.nodeActive.split(' ')[0] : "bg-slate-300")} />
                                <span className="text-xs text-slate-400 font-bold uppercase tracking-wider">Phase</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Nodes Stepper */}
                <div className="flex items-center justify-between relative px-4 pb-2">
                    {/* Connecting Line */}
                    <div className="absolute top-1/2 left-6 right-6 h-0.5 bg-slate-100 -z-0" />
                    
                    {nodes.map((node) => {
                        const isNodeActive = activeNodeId === node.id;
                        
                        return (
                            <div key={node.id} className="relative z-10 flex flex-col items-center gap-3 group/node cursor-pointer">
                                <motion.div 
                                    className={cn(
                                        "w-10 h-10 rounded-full flex items-center justify-center border-4 transition-all duration-300 bg-white shadow-sm relative",
                                        isNodeActive ? `${theme.nodeActive} shadow-lg scale-110` : "border-slate-100 text-slate-300"
                                    )}
                                    whileHover={{ scale: 1.1 }}
                                >
                                    {isNodeActive && (
                                        <span className="absolute inset-0 rounded-full animate-ping opacity-75 bg-current" />
                                    )}
                                    <div className={cn("w-2 h-2 rounded-full relative z-10", isNodeActive ? "bg-white" : "bg-slate-300")} />
                                </motion.div>
                                <div className="flex flex-col items-center">
                                    <span className={cn(
                                        "text-[10px] font-bold uppercase tracking-wider transition-colors px-2 py-1 rounded-md",
                                        isNodeActive ? `${theme.iconBg} ${theme.text}` : "text-slate-400"
                                    )}>
                                        {node.label}
                                    </span>
                                    {isNodeActive && (
                                        <motion.span 
                                            initial={{ opacity: 0, y: -5 }} 
                                            animate={{ opacity: 1, y: 0 }}
                                            className="text-[9px] text-slate-400 font-medium mt-0.5 flex items-center gap-1"
                                        >
                                            <Loader2 className="w-2 h-2 animate-spin" /> Running
                                        </motion.span>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Content / Config Area */}
            <div className="p-8 bg-white flex-1 relative">
                {isBlockActive && (
                    <div className={cn("absolute top-0 left-0 w-1 h-full", theme.iconBg)} />
                )}
                {children}
            </div>
        </motion.div>
    );
};

// --- Summary Panel Component ---
export const SummaryPanel = ({ state, sidebarWidth, chatWidth }: { 
    state: WorkflowState | null, 
    sidebarWidth: number, 
    chatWidth: number
}) => {
    const [isOpen, setIsOpen] = React.useState(true);

    if (!state) return null;

    return (
        <motion.div 
            initial={{ y: 100 }}
            animate={{ y: 0, left: sidebarWidth + 60, right: chatWidth + 0 }}
            className="fixed bottom-0 z-40 px-8 pb-0 pointer-events-none transition-all duration-300"
        >
            <div className="max-w-5xl mx-auto pointer-events-auto">
                <div className="bg-white/90 backdrop-blur-xl border border-slate-200 rounded-t-2xl shadow-[0_-10px_40px_-15px_rgba(0,0,0,0.1)] overflow-hidden ring-1 ring-slate-100">
                    <div 
                        className="h-10 bg-slate-50/50 border-b border-slate-100 flex items-center justify-between px-6 cursor-pointer hover:bg-slate-100 transition-colors"
                        onClick={() => setIsOpen(!isOpen)}
                    >
                        <div className="flex items-center gap-2 text-xs font-bold text-slate-500 uppercase tracking-wider">
                            <Database className="w-4 h-4" />
                            Evaluation Context
                        </div>
                        <Button variant="ghost" size="icon" className="h-6 w-6">
                            {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
                        </Button>
                    </div>

                    <AnimatePresence>
                        {isOpen && (
                            <motion.div 
                                initial={{ height: 0 }}
                                animate={{ height: "auto" }}
                                exit={{ height: 0 }}
                                className="overflow-hidden"
                            >
                                <div className="p-6 grid grid-cols-12 gap-8">
                                    <div className="col-span-12">
                                        <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-4 flex items-center gap-2">
                                             Selected Benchmarks
                                        </h4>
                                        <div className="flex gap-3 overflow-x-auto pb-2 scrollbar-hide">
                                            {state.benches?.length ? state.benches.map((b, i) => (
                                                <div key={i} className="min-w-[180px] bg-white rounded-xl p-3 border border-slate-200 shadow-sm flex flex-col gap-3 relative group hover:border-blue-400 hover:shadow-md transition-all">
                                                    <div className="flex justify-between items-start">
                                                        <span className="font-bold text-sm text-slate-800 line-clamp-1" title={b.bench_name}>{b.bench_name}</span>
                                                        <div className={cn(
                                                            "w-2 h-2 rounded-full ring-2 ring-white",
                                                            b.eval_status === "success" ? "bg-emerald-500" : "bg-slate-200"
                                                        )} />
                                                    </div>
                                                    
                                                    {b.meta?.eval_result ? (
                                                        <div className="mt-auto pt-2 border-t border-slate-50">
                                                            <div className="flex justify-between items-end">
                                                                <span className="text-[10px] text-slate-400 uppercase font-bold">Result</span>
                                                                <div className="text-right">
                                                                     {/* Parse and show metric if available */}
                                                                     {(() => {
                                                                         const res = b.meta.eval_result;
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
                                                                         return <span className="font-bold text-emerald-600 text-lg leading-none">{score === null ? "N/A" : (typeof score === 'number' ? score.toFixed(2) : String(score))}</span>
                                                                     })()}
                                                                </div>
                                                            </div>
                                                        </div>
                                                    ) : (
                                                        <div className="mt-auto h-6 flex items-end">
                                                            <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                                                                <div className="h-full bg-slate-300 w-1/3 rounded-full" />
                                                            </div>
                                                        </div>
                                                    )}
                                                </div>
                                            )) : (
                                                <div className="h-24 w-full border-2 border-dashed border-slate-100 rounded-xl flex items-center justify-center text-xs text-slate-400">
                                                    No benchmarks selected
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </motion.div>
    );
};

// --- Gallery Modal ---
export const GalleryModal = ({ isOpen, onClose, onSelect, apiBaseUrl }: { isOpen: boolean, onClose: () => void, onSelect: (bench: any) => void, apiBaseUrl: string }) => {
    const [benches, setBenches] = useState<any[]>([]);
    const [search, setSearch] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    useEffect(() => {
        if (isOpen) {
            setIsLoading(true);
            fetch(`${apiBaseUrl}/api/benches/gallery`)
                .then(res => res.json())
                .then(data => setBenches(data))
                .catch(err => console.error(err))
                .finally(() => setIsLoading(false));
        }
    }, [isOpen, apiBaseUrl]);

    const filtered = Array.isArray(benches) ? benches.filter(b => 
        b && ((b.bench_name || "").toLowerCase().includes(search.toLowerCase()) || 
        (b.description || "").toLowerCase().includes(search.toLowerCase()))
    ) : [];

    return (
        <Modal 
            isOpen={isOpen} 
            onClose={onClose} 
            title="Benchmark Gallery" 
            description="Select a benchmark to add to your evaluation."
        >
            <div className="space-y-4">
                <Input 
                    placeholder="Search benchmarks..." 
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    className="bg-slate-50"
                />
                
                {isLoading ? (
                    <div className="flex justify-center py-8"><Loader2 className="animate-spin text-slate-300" /></div>
                ) : (
                    <div className="grid grid-cols-1 gap-2 max-h-[400px] overflow-y-auto pr-2">
                        {filtered.map(b => (
                            <div key={b.bench_name} className="flex items-center justify-between p-3 rounded-lg border border-slate-100 hover:border-blue-200 hover:bg-blue-50/30 transition-all cursor-pointer group"
                                onClick={() => onSelect(b)}
                            >
                                <div>
                                    <div className="font-bold text-slate-700 text-sm flex items-center gap-2">
                                        {b.bench_name}
                                        {Array.isArray(b.task_type) && b.task_type.map((t: any, idx: number) => {
                                            const label = typeof t === 'object' && t !== null ? (t.name || JSON.stringify(t)) : String(t);
                                            return (
                                                <span key={idx} className="text-[10px] bg-slate-100 text-slate-500 px-1.5 rounded" title={typeof t === 'object' ? JSON.stringify(t) : undefined}>
                                                    {label}
                                                </span>
                                            );
                                        })}
                                    </div>
                                    <div className="text-xs text-slate-400 line-clamp-1">
                                        {typeof b.description === 'string' ? b.description : (b.description ? JSON.stringify(b.description) : '')}
                                    </div>
                                </div>
                                <Button size="sm" variant="ghost" className="opacity-0 group-hover:opacity-100">
                                    Add
                                </Button>
                            </div>
                        ))}
                        {filtered.length === 0 && <div className="text-center text-slate-400 py-4">No benchmarks found</div>}
                    </div>
                )}
            </div>
        </Modal>
    );
};
