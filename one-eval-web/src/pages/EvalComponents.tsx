import React, { useRef, useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
    Send, Check, Loader2, AlertCircle, ChevronDown, ChevronUp, 
    Database, Bot, Maximize2, X, Save as SaveIcon, Tag
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { Lang } from "@/lib/i18n";


import { ReportView } from "./ReportComponents";

// --- Types ---
export interface Bench {
  bench_name: string;
  eval_type?: string;
  bench_dataflow_eval_type?: string;
  meta?: any;
  eval_status?: string;
  download_status?: string;
  dataset_cache?: string;
}

export interface WorkflowState {
  user_query: string;
  benches: Bench[];
  target_model_name?: string;
  target_model?: any;
  metric_plan?: Record<string, any[]>;
  reports?: Record<string, any>;
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
export const BenchCard = ({ bench, activeNode, lang, onUpdate, onRetryDownload }: { bench: Bench, activeNode: string | null, lang: Lang, onUpdate?: (updatedBench: Bench) => void, onRetryDownload?: (params: { bench_name: string, config?: string, split?: string }) => Promise<void> }) => {
    const [isDetailsOpen, setIsDetailsOpen] = useState(false);
    const [isRetrying, setIsRetrying] = useState(false);
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
    const EVAL_TYPE_OPTIONS = [
        { value: "key1_text_score", label: tt("文本评分", "Text Scoring"), desc: tt("仅需文本字段", "Only text field required") },
        { value: "key2_qa", label: tt("问答单答案", "QA Single Answer"), desc: tt("问题+单参考答案", "Question + single reference") },
        { value: "key2_q_ma", label: tt("问答多答案", "QA Multi Answer"), desc: tt("问题+多参考答案", "Question + multiple references") },
        { value: "key3_q_choices_a", label: tt("单选题", "Multiple Choice"), desc: tt("题目+选项+正确选项", "Question + options + correct option") },
        { value: "key3_q_choices_as", label: tt("多选题", "Multi-select"), desc: tt("题目+选项+正确选项集合", "Question + options + labels set") },
        { value: "key3_q_a_rejected", label: tt("偏好对比", "Preference Pair"), desc: tt("题目+优选答案+拒绝答案", "Question + chosen + rejected") },
    ];
    const EVAL_TYPE_REQUIRED_KEYS: Record<string, string[]> = {
        key1_text_score: ["input_text_key"],
        key2_qa: ["input_question_key", "input_target_key"],
        key2_q_ma: ["input_question_key", "input_targets_key"],
        key3_q_choices_a: ["input_question_key", "input_choices_key", "input_label_key"],
        key3_q_choices_as: ["input_question_key", "input_choices_key", "input_labels_key"],
        key3_q_a_rejected: ["input_question_key", "input_better_key", "input_rejected_key"],
    };
    const EVAL_TYPE_GUIDE: Record<string, { scene: string; example: string }> = {
        key1_text_score: {
            scene: tt("纯文本质量评估（如困惑度）", "Pure text quality evaluation (e.g., perplexity)"),
            example: "text",
        },
        key2_qa: {
            scene: tt("问答任务，每题一个标准答案", "QA task with one reference answer"),
            example: "question, target",
        },
        key2_q_ma: {
            scene: tt("问答任务，每题多个可接受答案", "QA task with multiple acceptable answers"),
            example: "question, targets",
        },
        key3_q_choices_a: {
            scene: tt("单选题任务", "Single-choice task"),
            example: "question, choices, label",
        },
        key3_q_choices_as: {
            scene: tt("多选题任务", "Multi-select task"),
            example: "question, choices, labels",
        },
        key3_q_a_rejected: {
            scene: tt("偏好对比（好答案 vs 差答案）", "Preference pair (chosen vs rejected)"),
            example: "question, better, rejected",
        },
    };
    
    // Local state for editing in modal
    const [editKeyMap, setEditKeyMap] = useState<Record<string, string>>({});
    const [selectedSubset, setSelectedSubset] = useState<string>("");
    const [selectedSplit, setSelectedSplit] = useState<string>("");
    const [selectedEvalType, setSelectedEvalType] = useState<string>("");
    const [selectedPreviewRow, setSelectedPreviewRow] = useState<number>(0);
    
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
    const collapseMiddle = (raw: any, maxLen = 220, head = 120, tail = 80): string => {
        const text = typeof raw === "string" ? raw : JSON.stringify(raw);
        if (!text) return "";
        if (text.length <= maxLen) return text;
        return `${text.slice(0, head)} ... ${text.slice(-tail)}`;
    };

    // Extract Meta Data safely
    const meta = bench.meta || {};
    const structure = safeParse(meta.structure); 
    const keyMapping = safeParse(meta.key_mapping); 
    const downloadConfig = safeParse(meta.download_config); 
    const evalType = bench.bench_dataflow_eval_type || bench.eval_type || meta.bench_dataflow_eval_type; 
    const evalTypeLabel = (() => {
        const found = EVAL_TYPE_OPTIONS.find(x => x.value === (selectedEvalType || evalType));
        return found ? found.label : "";
    })();
    const descriptionRaw = meta.card_text ?? (lang === 'zh' ? (meta.description_zh || meta.description) : meta.description) ?? meta.desc ?? "No description available.";
    const description = typeof descriptionRaw === 'string' ? descriptionRaw : (descriptionRaw ? JSON.stringify(descriptionRaw) : "No description available.");
    const tags = Array.isArray(meta.tags) ? meta.tags : [];
    const availableKeys = Array.from(new Set([...(Array.isArray(meta.keys) ? meta.keys : []), ...(Array.isArray((bench as any).bench_keys) ? (bench as any).bench_keys : [])].map((x: any) => String(x))));
    const previewData = Array.isArray(meta.preview_data) ? meta.preview_data : []; // Preview rows
    const downloadPath = meta.download_path || meta.local_path || bench.dataset_cache;
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
    const keyMappingEntries = (() => {
        const required = EVAL_TYPE_REQUIRED_KEYS[selectedEvalType || String(evalType || "")] || [];
        const keys = Array.from(new Set([...required, ...Object.keys(editKeyMap || {})]));
        return keys;
    })();
    const keyOptionsListId = `key-options-${String(bench.bench_name || "bench").replace(/[^a-zA-Z0-9_-]/g, "_")}`;

    // Init state from bench
    useEffect(() => {
        if (!isDetailsOpen) return;
        if (keyMapping) {
            setEditKeyMap({ ...keyMapping });
        }
        setSelectedEvalType(typeof evalType === "string" ? evalType : "");
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
        setSelectedPreviewRow(0);
    }, [isDetailsOpen, bench.bench_name]);

    const handleSave = () => {
        if (!onUpdate) return;
        
        const updatedBench = { ...bench };
        if (!updatedBench.meta) updatedBench.meta = {};
        
        // Update Key Map
        updatedBench.meta.key_mapping = editKeyMap;
        updatedBench.bench_dataflow_eval_type = selectedEvalType || updatedBench.bench_dataflow_eval_type;
        updatedBench.meta.bench_dataflow_eval_type = selectedEvalType || updatedBench.meta.bench_dataflow_eval_type;
        
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

    const downloadError = typeof meta?.download_error === "string" ? meta.download_error : (meta?.download_error ? JSON.stringify(meta.download_error) : "");

    // Determine status color
    const statusColor = bench.download_status === "success" ? "bg-green-50 text-green-600" : "bg-slate-50 text-slate-400";
    
    // Check if we have meaningful data to show
    const hasData = structure || keyMapping || downloadConfig || evalType || (meta && Object.keys(meta).length > 0);
    const structureReady = !!structure?.ok;
    const structureFailed = !!meta?.structure_error;
    const downloadReady = bench.download_status === "success";
    const downloadFailed = bench.download_status === "failed";
    
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
                <div className="flex justify-between mb-2 shrink-0 gap-2">
                    <div className="flex flex-col gap-1 min-w-0">
                        <span className="text-sm font-bold text-slate-700 truncate pr-2" title={bench.bench_name}>{bench.bench_name}</span>
                        {meta.from_gallery === true ? (
                            <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-emerald-50 text-emerald-600 border border-emerald-100 w-fit">
                                ✓ Gallery · Ready to run
                            </span>
                        ) : meta.from_gallery === false ? (
                            <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-amber-50 text-amber-600 border border-amber-100 w-fit">
                                ⚠ HF Search · May need config
                            </span>
                        ) : null}
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                        {bench.download_status === "failed" && onRetryDownload && (
                            <Button
                                size="sm"
                                variant="outline"
                                className="h-6 px-2 text-[10px] gap-1"
                                disabled={isRetrying}
                                onClick={async (e) => {
                                    e.stopPropagation();
                                    setIsRetrying(true);
                                    try {
                                        const dl = bench.meta?.download_config || {};
                                        await onRetryDownload({ bench_name: bench.bench_name, config: dl.config, split: dl.split });
                                    } finally {
                                        setIsRetrying(false);
                                    }
                                }}
                            >
                                <Loader2 className={cn("w-3 h-3", isRetrying ? "animate-spin" : "")} />
                                {tt("重试", "Retry")}
                            </Button>
                        )}
                        <span className={cn(
                            "text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider h-fit whitespace-nowrap",
                            statusColor
                        )}>
                            {bench.download_status || tt("等待中", "Pending")}
                        </span>
                    </div>
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
                <div className="mb-3 flex items-center gap-2 text-[10px]">
                    <span className={cn(
                        "px-2 py-0.5 rounded-full border",
                        structureReady ? "bg-emerald-50 text-emerald-700 border-emerald-100" :
                        structureFailed ? "bg-red-50 text-red-700 border-red-100" :
                        "bg-amber-50 text-amber-700 border-amber-100"
                    )}>
                        {structureReady ? tt("结构已解析", "Structure Ready") : structureFailed ? tt("结构解析失败", "Structure Failed") : tt("结构解析中", "Parsing Structure")}
                    </span>
                    <span className={cn(
                        "px-2 py-0.5 rounded-full border",
                        downloadReady ? "bg-emerald-50 text-emerald-700 border-emerald-100" :
                        downloadFailed ? "bg-red-50 text-red-700 border-red-100" :
                        "bg-slate-50 text-slate-600 border-slate-100"
                    )}>
                        {downloadReady ? tt("下载完成", "Download Ready") : downloadFailed ? tt("下载失败", "Download Failed") : tt("等待下载", "Waiting Download")}
                    </span>
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
                                    <span className="text-xs italic">{tt("解析中...", "Analyzing...")}</span>
                                </>
                            ) : (
                                <span className="text-xs italic">{tt("等待解析...", "Waiting for analysis...")}</span>
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
                        description={tt("查看并修改该数据集配置。", "Review and modify the benchmark configuration.")}
                        footer={
                            <>
                                <Button variant="ghost" onClick={() => setIsDetailsOpen(false)}>{tt("取消", "Cancel")}</Button>
                                {bench.download_status === "failed" && onRetryDownload && (
                                    <Button
                                        variant="outline"
                                        disabled={isRetrying}
                                        onClick={async () => {
                                            setIsRetrying(true);
                                            try {
                                                await onRetryDownload({ bench_name: bench.bench_name, config: selectedSubset, split: selectedSplit });
                                            } finally {
                                                setIsRetrying(false);
                                            }
                                        }}
                                        className="gap-2"
                                    >
                                        <Loader2 className={cn("w-4 h-4", isRetrying ? "animate-spin" : "")} /> {tt("重试下载", "Retry Download")}
                                    </Button>
                                )}
                                <Button onClick={handleSave} className="bg-blue-600 hover:bg-blue-700 text-white gap-2">
                                    <SaveIcon className="w-4 h-4" /> {tt("保存配置", "Save Configuration")}
                                </Button>
                            </>
                        }
                    >
                        <div className="space-y-8">
                            {bench.download_status === "failed" && (
                                <div className="bg-red-50/50 p-4 rounded-xl border border-red-100">
                                    <div className="text-xs font-bold text-red-700 mb-1">{tt("下载失败", "Download Failed")}</div>
                                    {downloadError && <div className="text-xs text-red-600 font-mono whitespace-pre-wrap">{downloadError}</div>}
                                </div>
                            )}

                            {previewData.length > 0 && (
                                <div className="bg-slate-50 p-5 rounded-xl border border-slate-100">
                                    <h4 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-blue-500"/> {tt("数据预览", "Dataset Preview")}
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
                                                    <tr
                                                        key={idx}
                                                        className={cn("bg-white hover:bg-slate-50 cursor-pointer", selectedPreviewRow === idx ? "bg-blue-50/40" : "")}
                                                        onClick={() => setSelectedPreviewRow(idx)}
                                                    >
                                                        {isObject(row) ? Object.values(row).map((val: any, vi) => (
                                                            <td key={vi} className="px-3 py-2 font-mono text-slate-600 max-w-[220px] truncate border-r border-slate-50 last:border-r-0" title={String(val)}>
                                                                {collapseMiddle(val, 120, 55, 45)}
                                                            </td>
                                                        )) : <td className="px-3 py-2 text-slate-400 italic">{tt("无效数据行", "Invalid Row Data")}</td>}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                        <div className="bg-slate-50 px-3 py-1 text-[10px] text-slate-400 text-center border-t border-slate-200">
                                            {tt(`显示前 ${Math.min(previewData.length, 5)} 行，点击行查看完整内容`, `Showing first ${Math.min(previewData.length, 5)} rows, click row for full view`)}
                                        </div>
                                    </div>
                                    {isObject(previewData[selectedPreviewRow]) && (
                                        <div className="mt-3 p-3 rounded-lg border border-slate-200 bg-white">
                                            <div className="text-[11px] font-bold text-slate-500 mb-2">{tt(`第 ${selectedPreviewRow + 1} 行详情`, `Row ${selectedPreviewRow + 1} Details`)}</div>
                                            <div className="space-y-2 max-h-[260px] overflow-y-auto">
                                                {Object.entries(previewData[selectedPreviewRow]).map(([k, v]) => (
                                                    <div key={k} className="grid grid-cols-12 gap-2 text-xs">
                                                        <div className="col-span-4 text-slate-500 font-mono break-all">{k}</div>
                                                        <div className="col-span-8 text-slate-700 font-mono whitespace-pre-wrap break-all">{collapseMiddle(v, 1800, 1000, 600)}</div>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            <div className="bg-indigo-50/50 p-5 rounded-xl border border-indigo-100">
                                <div className="flex items-center justify-between mb-3">
                                    <h4 className="text-sm font-bold text-indigo-800 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-indigo-500"/> {tt("评测类型", "Evaluation Type")}
                                    </h4>
                                    <span className="text-[10px] text-indigo-500 uppercase font-bold tracking-wider">{tt("必填", "Required")}</span>
                                </div>
                                <select
                                    value={selectedEvalType}
                                    onChange={(e) => {
                                        const next = e.target.value;
                                        setSelectedEvalType(next);
                                        const required = EVAL_TYPE_REQUIRED_KEYS[next] || [];
                                        setEditKeyMap(prev => {
                                            const out = { ...prev };
                                            required.forEach(k => {
                                                if (!(k in out)) out[k] = "";
                                            });
                                            return out;
                                        });
                                    }}
                                    className="w-full h-9 rounded-lg border border-indigo-200 bg-white px-3 text-sm font-bold text-slate-900 shadow-sm focus:outline-none focus:ring-2 focus:ring-indigo-500/20 focus:border-indigo-500 transition-all"
                                >
                                    <option value="">{tt("请选择评测类型", "Select evaluation type")}</option>
                                    {EVAL_TYPE_OPTIONS.map(opt => (
                                        <option key={opt.value} value={opt.value}>{opt.label}</option>
                                    ))}
                                </select>
                                <div className="mt-2 text-xs text-indigo-700">
                                    {selectedEvalType
                                        ? (EVAL_TYPE_OPTIONS.find(x => x.value === selectedEvalType)?.desc || "")
                                        : tt("用于决定数据字段如何映射并驱动 DataFlow 评测。", "Defines field mapping and drives DataFlow evaluation.")}
                                </div>
                                {selectedEvalType && (
                                    <div className="mt-3 rounded-lg border border-indigo-100 bg-white p-3 text-xs text-slate-700 space-y-1">
                                        <div><b>{tt("适用场景", "Use case")}:</b> {EVAL_TYPE_GUIDE[selectedEvalType]?.scene}</div>
                                        <div><b>{tt("建议字段", "Suggested fields")}:</b> {EVAL_TYPE_GUIDE[selectedEvalType]?.example}</div>
                                        <div><b>{tt("必填映射", "Required mapping")}:</b> {(EVAL_TYPE_REQUIRED_KEYS[selectedEvalType] || []).join(", ")}</div>
                                    </div>
                                )}
                            </div>
                            
                            {/* 1. Key Mapping Section (Editable) */}
                                <div className="bg-amber-50/50 p-5 rounded-xl border border-amber-100">
                                    <div className="flex justify-between items-center mb-4">
                                        <h4 className="text-sm font-bold text-amber-800 flex items-center gap-2">
                                            <span className="w-2 h-2 rounded-full bg-amber-500"/> Key Mapping
                                        </h4>
                                        {(selectedEvalType || evalType) && (
                                            <div className="text-xs text-amber-700 bg-amber-100 px-2 py-1 rounded border border-amber-200">
                                                {tt("类型", "Type")}: <b>{evalTypeLabel || tt("未选择", "Not selected")}</b>
                                            </div>
                                        )}
                                    </div>
                                    <datalist id={keyOptionsListId}>
                                        {availableKeys.map((ak: any) => (
                                            <option key={String(ak)} value={String(ak)} />
                                        ))}
                                    </datalist>
                                    <div className="grid grid-cols-2 gap-x-8 gap-y-4 text-sm">
                                        {keyMappingEntries.map((k) => (
                                            <div key={k} className="flex items-center gap-3">
                                                <span className="text-slate-500 font-mono w-1/3 text-right text-xs truncate" title={k}>{k}</span>
                                                <Input 
                                                    list={keyOptionsListId}
                                                    value={String((editKeyMap as any)[k] ?? "")}
                                                    onChange={(e) => setEditKeyMap({ ...editKeyMap, [k]: e.target.value })}
                                                    placeholder={tt("支持嵌套路径，如 mc1_targets.choices", "Nested path supported, e.g. mc1_targets.choices")}
                                                    className="h-8 bg-white border-amber-200 focus-visible:ring-amber-500 font-mono text-xs font-bold text-slate-800"
                                                />
                                            </div>
                                        ))}
                                    </div>
                                </div>

                            {/* Local Cache Path */}
                            {downloadPath && (
                                <div className="bg-slate-50 p-5 rounded-xl border border-slate-100">
                                    <h4 className="text-sm font-bold text-slate-700 mb-3 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-slate-500"/> {tt("数据集本地存储位置", "Dataset Local Storage Path")}
                                    </h4>
                                    <div className="p-3 rounded-lg border border-slate-200 bg-white text-xs font-mono text-slate-600 break-all">
                                        {downloadPath}
                                    </div>
                                </div>
                            )}


                            {/* 2. Config Selector Section */}
                            {structureSubsets.length > 0 && (
                                <div className="bg-slate-50 p-5 rounded-xl border border-slate-100">
                                    <h4 className="text-sm font-bold text-slate-700 mb-4 flex items-center gap-2">
                                        <span className="w-2 h-2 rounded-full bg-slate-500"/> {tt("数据集配置", "Dataset Configuration")}
                                    </h4>
                                    
                                    <div className="space-y-4">
                                        {/* Subsets */}
                                        <div>
                                            <label className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2 block">{tt("子集 / 配置", "Subset / Config")}</label>
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
                                        <span className="w-2 h-2 rounded-full bg-blue-500"/> {tt("最终配置", "Final Configuration")}
                                    </h4>
                                    <span className="text-[10px] text-blue-400 uppercase font-bold tracking-wider">
                                        {tt("推荐配置", "Agent Recommended Configuration")}
                                    </span>
                                </div>
                                <div className="bg-white p-3 rounded-lg border border-blue-100 flex gap-4 items-center">
                                    <div className="flex-1">
                                        <div className="text-[10px] text-slate-400 uppercase">Config</div>
                                        <div className="font-mono font-bold text-blue-700">{selectedSubset || tt("未选择", "Not selected")}</div>
                                    </div>
                                    <div className="w-px h-8 bg-blue-100" />
                                    <div className="flex-1">
                                        <div className="text-[10px] text-slate-400 uppercase">Split</div>
                                        <div className="font-mono font-bold text-blue-700">{selectedSplit || tt("未选择", "Not selected")}</div>
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
                                <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">{tt("原始元数据", "Raw Metadata")}</h4>
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
    onStop?: () => void;
    isWaitingForInput: boolean;
    activeNodeId?: string | null;
    isCollapsed: boolean;
    onToggleCollapse: () => void;
    lang: Lang;
    interruptToken?: string | null;
}

const EMOJIS = ["✨", "🤖", "🚀", "💡", "🔮", "✅", "🎯"];

export const ChatPanel = ({ messages, status, onSendMessage, onConfirm, onStop, isWaitingForInput, activeNodeId, isCollapsed, onToggleCollapse, lang, interruptToken }: ChatPanelProps) => {
    const [input, setInput] = React.useState("");
    const [dismissedInterrupts, setDismissedInterrupts] = React.useState<string[]>([]);
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
    const confirmText = activeNodeId?.includes("PreEvalReviewNode")
        ? tt("请先在执行阶段检查目标模型与评测参数，确认无误后点击批准开始评测。", "Please review the target model and evaluation parameters in the Execution Phase, then approve to start evaluation.")
        : tt("我已准备好基准配置，请检查流程块中的高亮参数。", "I've prepared the benchmark configuration. Please review the highlighted parameters in the workflow blocks.");
    
    const handleConfirm = () => {
        if (interruptToken) {
            setDismissedInterrupts(prev => prev.includes(interruptToken) ? prev : [...prev, interruptToken]);
        }
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
    
    const getEmojiForMessage = (messageId: string) => {
        let hash = 0;
        for (let i = 0; i < messageId.length; i++) {
            hash = (hash * 31 + messageId.charCodeAt(i)) >>> 0;
        }
        return EMOJIS[hash % EMOJIS.length];
    };

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
                            <span>{tt("OneEval 助手", "OneEval Assistant")}</span>
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
                                    {msg.role === "ai" && <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{tt("AI 助手", "AI Assistant")}</span>}
                                    {msg.role === "user" && <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{tt("你", "You")}</span>}
                                </div>
                                
                                <div className={cn(
                                    "px-5 py-3.5 text-sm shadow-sm leading-relaxed",
                                    msg.role === "user" 
                                        ? "bg-gradient-to-br from-blue-600 to-indigo-600 text-white rounded-2xl rounded-tr-sm shadow-blue-500/20" 
                                        : "bg-white border border-white/60 text-slate-700 rounded-2xl rounded-tl-sm shadow-sm"
                                )}>
                                    {msg.role === "ai" && <span className="mr-2">{getEmojiForMessage(msg.id)}</span>}
                                    {msg.content}
                                </div>
                                <span className="text-[10px] text-slate-300 mt-1 px-1">
                                    {new Date(msg.timestamp).toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}
                                </span>
                            </motion.div>
                        ))}

                        {/* Confirmation Card */}
                        {/* Plan C: Also show approve button when activeNodeId is an interrupt node (handles race condition in RAG mode) */}
                        {status === "interrupted" && interruptToken && !dismissedInterrupts.includes(interruptToken) && (
                            <motion.div 
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                className="self-start w-full pr-8"
                            >
                                <div className="bg-gradient-to-br from-white to-amber-50/50 border border-amber-100 rounded-2xl p-5 shadow-lg shadow-amber-500/5 ring-1 ring-amber-100/50 relative overflow-hidden">
                                    <div className="absolute top-0 right-0 w-16 h-16 bg-amber-500/10 rounded-full blur-2xl -mr-8 -mt-8" />
                                    
                                    <div className="flex items-center gap-2 mb-3 text-amber-600 font-bold text-sm">
                                        <AlertCircle className="w-4 h-4" />
                                        {tt("需要确认", "Review Required")}
                                    </div>
                                    <p className="text-sm text-slate-600 mb-5 leading-relaxed">
                                        {confirmText}
                                    </p>
                                    <div className="flex gap-3">
                                        <Button size="sm" onClick={handleConfirm} className="flex-1 bg-amber-500 hover:bg-amber-600 text-white shadow-lg shadow-amber-500/20 border-0 rounded-xl h-9">
                                            <Check className="w-4 h-4 mr-1.5" /> {tt("批准", "Approve")}
                                        </Button>
                                        <Button size="sm" variant="outline" onClick={onStop} className="h-9 rounded-xl">
                                            {tt("停止", "Stop")}
                                        </Button>
                                    </div>
                                </div>
                            </motion.div>
                        )}
                        
                        {status === "running" && (
                             <div className="self-start flex items-center gap-2 text-xs text-slate-400 pl-2 bg-slate-50/50 px-3 py-1.5 rounded-full border border-slate-100">
                                 <Loader2 className="w-3 h-3 animate-spin text-blue-500" /> 
                                 <span>{tt("工作流执行中...", "Processing workflow...")}</span>
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
                                    placeholder={tt("输入消息...", "Type a message...")}
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
                            {tt("按 Enter 发送", "Press Enter to send")}
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
    lang: Lang;
    children: React.ReactNode;
}

export const WorkflowBlock = ({ title, icon: Icon, nodes, activeNodeId, status, colorTheme, lang, children }: WorkflowBlockProps) => {
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
    const isBlockActive = nodes.some(n => n.id === activeNodeId) || status === "completed" || status === "interrupted" || status === "running"; 
    
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
                                <span className="text-xs text-slate-400 font-bold uppercase tracking-wider">{lang === "zh" ? "阶段" : "Phase"}</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Nodes Stepper */}
                <div className="flex items-center justify-between relative px-4 pb-2">
                    {/* Connecting Line */}
                    <div className="absolute top-1/2 left-6 right-6 h-0.5 bg-slate-100 -z-0" />
                    
                    {nodes.map((node, index) => {
                        const isNodeActive = activeNodeId === node.id;
                        // Determine if node is "passed" (completed)
                        // A simple heuristic: if activeNodeId is later in the list than this node
                        const activeIndex = nodes.findIndex(n => n.id === activeNodeId);
                        const isPassed = activeIndex > -1 && index < activeIndex;
                        const isCompletedStatus = status === "completed";

                        return (
                            <div key={node.id} className="relative z-10 flex flex-col items-center gap-3 group/node cursor-pointer">
                                <motion.div 
                                    className={cn(
                                        "w-10 h-10 rounded-full flex items-center justify-center border-4 transition-all duration-300 bg-white shadow-sm relative",
                                        isNodeActive 
                                            ? `${theme.nodeActive} shadow-lg scale-110` 
                                            : (isPassed || isCompletedStatus) 
                                                ? `${theme.text.replace('text-', 'border-')} ${theme.text}` 
                                                : "border-slate-100 text-slate-300"
                                    )}
                                    whileHover={{ scale: 1.1 }}
                                >
                                    {isNodeActive && (
                                        <span className="absolute inset-0 rounded-full animate-ping opacity-75 bg-current" />
                                    )}
                                    {(isPassed || isCompletedStatus) && !isNodeActive && (
                                         <div className={cn("w-2 h-2 rounded-full relative z-10", theme.nodeActive.split(' ')[0])} />
                                    )}
                                    <div className={cn("w-2 h-2 rounded-full relative z-10", isNodeActive ? "bg-white" : ((isPassed || isCompletedStatus) ? "hidden" : "bg-slate-300"))} />
                                </motion.div>
                                <div className="flex flex-col items-center">
                                    <span className={cn(
                                        "text-[10px] font-bold uppercase tracking-wider transition-colors px-2 py-1 rounded-md",
                                        isNodeActive ? `${theme.iconBg} ${theme.text}` : ((isPassed || isCompletedStatus) ? theme.text : "text-slate-400")
                                    )}>
                                        {node.label}
                                    </span>
                                    {isNodeActive && (
                                        <motion.span 
                                            initial={{ opacity: 0, y: -5 }} 
                                            animate={{ opacity: 1, y: 0 }}
                                            className="text-[9px] text-slate-400 font-medium mt-0.5 flex items-center gap-1"
                                        >
                                            <Loader2 className="w-2 h-2 animate-spin" /> {lang === "zh" ? "运行中" : "Running"}
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



// --- Summary Bench Card Component ---
const SummaryBenchCard = ({ bench, lang }: { bench: any, lang: Lang }) => {
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
    const [isResultOpen, setIsResultOpen] = useState(true);
    // const [isSummaryOpen, setIsSummaryOpen] = useState(false);
    
    const hasResult = bench.meta?.eval_result && Object.keys(bench.meta.eval_result).length > 0;
    // const hasSummary = !!bench.meta?.metric_summary;

    return (
        <div className="min-w-[320px] bg-white rounded-xl p-3 border border-slate-200 shadow-sm flex flex-col gap-3 relative group hover:border-blue-400 hover:shadow-md transition-all self-start">
            <div className="flex justify-between items-start">
                <span className="font-bold text-sm text-slate-800 line-clamp-1" title={bench.bench_name}>{bench.bench_name}</span>
                <div className={cn(
                    "w-2 h-2 rounded-full ring-2 ring-white shrink-0 ml-2",
                    bench.eval_status === "success" ? "bg-emerald-500" : "bg-slate-200"
                )} />
            </div>
            
            <div className="space-y-2">
                {/* Metric Result Section */}
                {hasResult ? (
                    <div className="border border-slate-100 rounded-lg overflow-hidden bg-slate-50/30">
                        <div 
                            className="bg-slate-50 px-3 py-2 flex justify-between items-center cursor-pointer hover:bg-slate-100 transition-colors"
                            onClick={() => setIsResultOpen(!isResultOpen)}
                        >
                            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1">
                                <Tag className="w-3 h-3" /> {tt("指标结果", "Metric Result")}
                            </span>
                            {isResultOpen ? <ChevronUp className="w-3 h-3 text-slate-400"/> : <ChevronDown className="w-3 h-3 text-slate-400"/>}
                        </div>
                        {isResultOpen && (
                             <div className="p-3 bg-white text-xs space-y-1 border-t border-slate-100">
                                 {Object.entries(bench.meta.eval_result).map(([k, v]) => (
                                     <div key={k} className="flex justify-between items-center border-b border-slate-50 last:border-0 pb-1 last:pb-0">
                                         <span className="text-slate-500 font-medium truncate pr-2" title={k}>{k}</span>
                                         <span className="font-mono font-bold text-emerald-600">
                                            {typeof v === 'number' ? v.toFixed(4) : String(v)}
                                         </span>
                                     </div>
                                 ))}
                             </div>
                        )}
                    </div>
                ) : (
                    <div className="h-10 flex items-center justify-center text-[10px] text-slate-300 italic border border-dashed border-slate-100 rounded-lg">
                        {tt("暂无结果", "No results yet")}
                    </div>
                )}

                {/* Metric Summary Section */}
                {/* {hasSummary && (
                    <div className="border border-slate-100 rounded-lg overflow-hidden bg-slate-50/30">
                        <div 
                            className="bg-slate-50 px-3 py-2 flex justify-between items-center cursor-pointer hover:bg-slate-100 transition-colors"
                            onClick={() => setIsSummaryOpen(!isSummaryOpen)}
                        >
                            <span className="text-[10px] font-bold text-slate-500 uppercase tracking-wider flex items-center gap-1">
                                <Bot className="w-3 h-3" /> Metric Summary
                            </span>
                            {isSummaryOpen ? <ChevronUp className="w-3 h-3 text-slate-400"/> : <ChevronDown className="w-3 h-3 text-slate-400"/>}
                        </div>
                        {isSummaryOpen && (
                             <div className="p-3 bg-white border-t border-slate-100 max-h-[300px] overflow-y-auto scrollbar-thin scrollbar-thumb-slate-200">
                                 <SimpleMarkdown content={bench.meta.metric_summary} />
                             </div>
                        )}
                    </div>
                )} */}
            </div>
        </div>
    );
};

// --- Summary Panel Component ---
export const SummaryPanel = ({ state, sidebarWidth, chatWidth, lang }: { 
    state: WorkflowState | null, 
    sidebarWidth: number, 
    chatWidth: number,
    lang: Lang
}) => {
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
    const [isOpen, setIsOpen] = React.useState(true);
    const [viewMode, setViewMode] = React.useState<"benches" | "report">("benches");

    if (!state) return null;

    const hasReport = state.reports && state.reports["default"];

    // Auto-switch to report view when available and not manually switched
    // useEffect(() => {
    //     if (hasReport) {
    //         setViewMode("report");
    //         setIsOpen(true);
    //     }
    // }, [hasReport]);

    return (
        <motion.div 
            initial={{ y: 100 }}
            animate={{ 
                y: 0, 
                left: sidebarWidth + 60, 
                right: chatWidth + 0,
                height: isOpen ? (viewMode === "report" ? "85vh" : "auto") : "auto" 
            }}
            className="fixed bottom-0 z-40 px-8 pb-0 pointer-events-none transition-all duration-300"
        >
            <div className="max-w-6xl mx-auto pointer-events-auto h-full flex flex-col justify-end">
                <div className="bg-white/90 backdrop-blur-xl border border-slate-200 rounded-t-2xl shadow-[0_-10px_40px_-15px_rgba(0,0,0,0.1)] overflow-hidden ring-1 ring-slate-100 flex flex-col max-h-full transition-all duration-300">
                    <div 
                        className="h-12 bg-slate-50/50 border-b border-slate-100 flex items-center justify-between px-6 shrink-0"
                    >
                        <div className="flex items-center gap-6 h-full">
                             <div 
                                className={cn(
                                    "flex items-center gap-2 text-xs font-bold uppercase tracking-wider h-full border-b-2 transition-all cursor-pointer px-2",
                                    viewMode === "benches" ? "text-slate-600 border-slate-600" : "text-slate-400 border-transparent hover:text-slate-500"
                                )}
                                onClick={() => { setViewMode("benches"); setIsOpen(true); }}
                            >
                                <Database className="w-4 h-4" />
                                {tt("上下文", "Context")}
                            </div>
                            
                            {hasReport && (
                                <div 
                                    className={cn(
                                        "flex items-center gap-2 text-xs font-bold uppercase tracking-wider h-full border-b-2 transition-all cursor-pointer px-2 relative",
                                        viewMode === "report" ? "text-violet-600 border-violet-600 bg-violet-50/50" : "text-slate-400 border-transparent hover:text-slate-500"
                                    )}
                                    onClick={() => { setViewMode("report"); setIsOpen(true); }}
                                >
                                    <Bot className="w-4 h-4" />
                                    {tt("最终报告", "Final Report")}
                                    <span className="absolute top-3 right-0 w-1.5 h-1.5 bg-violet-500 rounded-full animate-pulse" />
                                </div>
                            )}
                        </div>

                        <div 
                            className="flex items-center gap-2 cursor-pointer p-2 hover:bg-slate-100 rounded-lg transition-colors"
                            onClick={() => setIsOpen(!isOpen)}
                        >
                            <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                                {isOpen ? tt("收起", "Collapse") : tt("展开", "Expand")}
                            </span>
                            {isOpen ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronUp className="w-4 h-4 text-slate-400" />}
                        </div>
                    </div>

                    <AnimatePresence mode="wait">
                        {isOpen && (
                            <motion.div 
                                key={viewMode}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 0.2 }}
                                className="overflow-hidden flex-1 flex flex-col bg-slate-50/30"
                            >
                                {viewMode === "benches" ? (
                                    <div className="p-6 overflow-x-auto">
                                        <div className="flex gap-4 pb-4 scrollbar-hide items-start min-h-[120px]">
                                            {state.benches?.length ? state.benches.map((b, i) => (
                                                <SummaryBenchCard key={i} bench={b} lang={lang} />
                                            )) : (
                                                <div className="h-24 w-full border-2 border-dashed border-slate-100 rounded-xl flex items-center justify-center text-xs text-slate-400">
                                                    {tt("暂无已选基准", "No benchmarks selected")}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                ) : (
                                    <div className="p-8 overflow-y-auto flex-1 min-h-[400px] scrollbar-thin scrollbar-thumb-slate-200">
                                        <div className="max-w-5xl mx-auto">
                                            <ReportView report={state.reports?.["default"]} lang={lang} />
                                        </div>
                                    </div>
                                )}
                            </motion.div>
                        )}
                    </AnimatePresence>
                </div>
            </div>
        </motion.div>
    );
};

// --- Gallery Modal ---
// 适配 bench_gallery.json 数据结构
export const GalleryModal = ({ isOpen, onClose, onSelect, apiBaseUrl, lang }: { isOpen: boolean, onClose: () => void, onSelect: (bench: any) => void, apiBaseUrl: string, lang: Lang }) => {
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
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

    // 从 bench_gallery.json 获取描述和标签
    const getDescription = (b: any): string => {
        // 优先从 meta.description 获取，中文模式下优先使用 description_zh
        if (lang === 'zh' && b.meta?.description_zh) return b.meta.description_zh;
        if (b.meta?.description) return b.meta.description;
        // 兼容旧格式
        if (typeof b.description === 'string') return b.description;
        return '';
    };

    const getTags = (b: any): string[] => {
        // 优先从 meta.tags 获取
        if (Array.isArray(b.meta?.tags)) return b.meta.tags;
        // 兼容旧格式 task_type
        if (Array.isArray(b.task_type)) return b.task_type;
        return [];
    };

    const getDisplayName = (b: any): string => {
        // 优先从 meta.aliases 获取显示名称
        if (Array.isArray(b.meta?.aliases) && b.meta.aliases.length > 1) {
            return b.meta.aliases[1]; // 第二个通常是大写版本
        }
        return b.bench_name || '';
    };

    const getCategory = (b: any): string => {
        return b.meta?.category || '';
    };

    const filtered = Array.isArray(benches) ? benches.filter(b => {
        if (!b) return false;
        const searchLower = search.toLowerCase();
        const benchName = (b.bench_name || "").toLowerCase();
        const description = getDescription(b).toLowerCase();
        const tags = getTags(b).join(" ").toLowerCase();
        const category = getCategory(b).toLowerCase();
        return benchName.includes(searchLower) ||
               description.includes(searchLower) ||
               tags.includes(searchLower) ||
               category.includes(searchLower);
    }) : [];

    return (
        <Modal
            isOpen={isOpen}
            onClose={onClose}
            title={tt("基准库", "Benchmark Gallery")}
            description={tt("选择一个基准并加入当前评测。", "Select a benchmark to add to your evaluation.")}
        >
            <div className="space-y-4">
                <Input
                    placeholder={tt("搜索基准...", "Search benchmarks...")}
                    value={search}
                    onChange={e => setSearch(e.target.value)}
                    className="bg-slate-50"
                />

                {isLoading ? (
                    <div className="flex justify-center py-8"><Loader2 className="animate-spin text-slate-300" /></div>
                ) : (
                    <div className="grid grid-cols-1 gap-2 max-h-[400px] overflow-y-auto pr-2">
                        {filtered.map(b => {
                            const tags = getTags(b);
                            const category = getCategory(b);
                            return (
                                <div key={b.bench_name} className="flex items-center justify-between p-3 rounded-lg border border-slate-100 hover:border-blue-200 hover:bg-blue-50/30 transition-all cursor-pointer group"
                                    onClick={() => onSelect(b)}
                                >
                                    <div className="flex-1 min-w-0">
                                        <div className="font-bold text-slate-700 text-sm flex items-center gap-2 flex-wrap">
                                            {getDisplayName(b)}
                                            {category && (
                                                <span className="text-[10px] bg-blue-100 text-blue-600 px-1.5 rounded">
                                                    {category}
                                                </span>
                                            )}
                                            {tags.slice(0, 2).map((t: string, idx: number) => (
                                                <span key={idx} className="text-[10px] bg-slate-100 text-slate-500 px-1.5 rounded">
                                                    {t}
                                                </span>
                                            ))}
                                            {tags.length > 2 && (
                                                <span className="text-[10px] bg-slate-100 text-slate-400 px-1.5 rounded">
                                                    +{tags.length - 2}
                                                </span>
                                            )}
                                        </div>
                                        <div className="text-xs text-slate-400 line-clamp-1 mt-0.5">
                                            {getDescription(b)}
                                        </div>
                                    </div>
                                    <Button size="sm" variant="ghost" className="opacity-0 group-hover:opacity-100 shrink-0 ml-2">
                                        {tt("添加", "Add")}
                                    </Button>
                                </div>
                            );
                        })}
                        {filtered.length === 0 && <div className="text-center text-slate-400 py-4">{tt("未找到匹配基准", "No benchmarks found")}</div>}
                    </div>
                )}
            </div>
        </Modal>
    );
};
