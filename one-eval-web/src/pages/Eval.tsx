import { useState, useEffect, useMemo } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { 
    Clock, X, Search, Database, Play, Save, Layers, Plus, BookOpen, Trash2, AlertTriangle, Settings, ChevronRight, ChevronDown, Check, RefreshCw, Bot, Tag, ChevronUp
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { ChatPanel, WorkflowBlock, SummaryPanel, Bench, WorkflowState, BenchCard, GalleryModal } from "./EvalComponents";
import { SimpleMarkdown } from "@/components/ui/simple-markdown";
import { useLang } from "@/lib/i18n";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// --- Types ---
interface StatusResponse {
  thread_id: string;
  status: "idle" | "running" | "interrupted" | "completed" | "failed" | "not_found" | "stopped";
  next_node: string[] | null;
  state_values: WorkflowState | null;
  current_node?: string; 
  interrupts?: Array<{ value?: unknown }>;
  eval_progress?: {
    bench_name?: string;
    stage?: string;
    generated?: number;
    total?: number;
    percent?: number;
  } | null;
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

interface MetricMeta {
    name: string;
    desc: string;
    usage: string;
    categories: string[];
    aliases: string[];
}

export const Eval = () => {
  const { lang, setLang, t } = useLang();
    const [metricRegistry, setMetricRegistry] = useState<MetricMeta[]>([]);
  const [workMode, setWorkMode] = useState<"agent" | "manual">(() => {
      const v = localStorage.getItem("oneEval.workMode");
      return v === "manual" ? "manual" : "agent";
  });
  const [query, setQuery] = useState("");
  const [threadId, setThreadId] = useState<string | null>(null);
  const [status, setStatus] = useState<StatusResponse["status"]>("idle");
  const [state, setState] = useState<WorkflowState | null>(null);
  const [currentNode, setCurrentNode] = useState<string | null>(null);
  const [interruptToken, setInterruptToken] = useState<string | null>(null);
  const [evalProgress, setEvalProgress] = useState<StatusResponse["eval_progress"]>(null);
  
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
      top_k: -1,
      repetition_penalty: 1.0,
      max_tokens: 2048,
      tensor_parallel_size: 1,
      max_model_len: 32768,
      gpu_memory_utilization: 0.9,
      seed: 0
  });

  const [expandedResults, setExpandedResults] = useState<number[]>([]);
  const [expandedMetricResults, setExpandedMetricResults] = useState<string[]>([]);
  const [expandedMetricSummaries, setExpandedMetricSummaries] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<any | null>(null);
  const [manualModelPath, setManualModelPath] = useState<string>("");
  const [manualBenches, setManualBenches] = useState<Bench[]>([]);
  const [editMetricPlan, setEditMetricPlan] = useState<Record<string, any[]> | null>(null);
  const [addingMetricBench, setAddingMetricBench] = useState<string | null>(null);
  const [metricSearch, setMetricSearch] = useState("");

  const handleAddMetric = (benchName: string, metric: MetricMeta) => {
      const newPlan = { ...(editMetricPlan || state?.metric_plan || {}) };
      // Deep copy to avoid mutation
      Object.keys(newPlan).forEach(k => {
          if (!Array.isArray(newPlan[k])) newPlan[k] = [];
          else newPlan[k] = [...newPlan[k]];
      });
      
      if (!newPlan[benchName]) newPlan[benchName] = [];
      
      // Check duplicate
      if (newPlan[benchName].some((m: any) => m.name === metric.name)) {
          return;
      }
      
      newPlan[benchName].push({
          name: metric.name,
          priority: "secondary", // Default
          desc: metric.desc,
          args: {}
      });
      
      setEditMetricPlan(newPlan);
      setAddingMetricBench(null);
      setMetricSearch("");
  };
  
  // Chat State
  const [messages, setMessages] = useState<ChatMessage[]>([
      { id: "init", role: "ai", content: t({ zh: "你好！请先描述你的评测目标，我们将自动开始流程。", en: "Hello! Describe your evaluation task to get started." }), timestamp: Date.now() }
  ]);

  useEffect(() => {
      setMessages(prev => prev.map(m => m.id === "init"
        ? { ...m, content: t({ zh: "你好！请先描述你的评测目标，我们将自动开始流程。", en: "Hello! Describe your evaluation task to get started." }) }
        : m
      ));
  }, [lang]);

  // Editable State (for manual modification)
  const [editBenches, setEditBenches] = useState<Bench[]>([]);
  const [availableModels, setAvailableModels] = useState<any[]>([]);
  const [useRAG, setUseRAG] = useState(true);
  const [localCount, setLocalCount] = useState(3);
  const [hfCount, setHfCount] = useState(2);

  const apiBaseUrl = useMemo(() => localStorage.getItem("oneEval.apiBaseUrl") || "http://localhost:8000", []);
  const draftKey = useMemo(() => "oneEval.evalDraft", []);

  useEffect(() => {
      localStorage.setItem("oneEval.workMode", workMode);
  }, [workMode]);

  useEffect(() => {
      if (threadId) return;
      try {
          const raw = localStorage.getItem(draftKey);
          if (!raw) return;
          const draft = JSON.parse(raw);
          if (draft?.query) setQuery(draft.query);
          if (draft?.manualModelPath) setManualModelPath(draft.manualModelPath);
          if (Array.isArray(draft?.manualBenches)) setManualBenches(draft.manualBenches);
          if (Array.isArray(draft?.editBenches)) setEditBenches(draft.editBenches);
          if (draft?.evalParams && typeof draft.evalParams === "object") {
              setEvalParams(prev => ({ ...prev, ...draft.evalParams }));
          }
          if (draft?.workMode === "manual" || draft?.workMode === "agent") setWorkMode(draft.workMode);
          if (typeof draft?.useRAG === "boolean") setUseRAG(draft.useRAG);
          if (typeof draft?.localCount === "number") setLocalCount(draft.localCount);
          if (typeof draft?.hfCount === "number") setHfCount(draft.hfCount);
      } catch {}
  }, [draftKey, threadId]);

  useEffect(() => {
      if (selectedModel?.path && !manualModelPath) {
          setManualModelPath(selectedModel.path);
      }
  }, [selectedModel?.path]);
  
  // Fetch Models
  useEffect(() => {
      axios.get(`${apiBaseUrl}/api/models`)
          .then(res => {
              if (Array.isArray(res.data)) {
                  setAvailableModels(res.data);
              }
          })
          .catch(e => console.error("Failed to fetch models", e));

      // Fetch Metrics Registry
      axios.get(`${apiBaseUrl}/api/metrics/registry`)
          .then(res => {
              if (Array.isArray(res.data)) {
                  setMetricRegistry(res.data);
              }
          })
          .catch(e => console.error("Failed to fetch metrics registry", e));
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
    if (!threadId || status === "completed" || status === "failed" || status === "stopped" || isResuming) return;

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
                            const failText = t({ zh: "评测失败，请查看日志。", en: "Evaluation failed. Please check logs." });
                            if (lastMsg?.content !== failText) {
                                 return [...prev, { id: Date.now().toString(), role: "system", content: failText, timestamp: Date.now() }];
                            }
                        }
                        return prev;
                    });
                }

        setStatus(data.status);
        setEvalProgress(data.eval_progress ?? null);
        if (data.status === "interrupted") {
            const interruptValue = data.interrupts?.[0]?.value;
            const token = `${threadId || ""}|${data.next_node?.[0] || ""}|${JSON.stringify(interruptValue ?? "")}`;
            setInterruptToken(token);
        } else {
            setInterruptToken(null);
        }
        if (data.state_values) {
            setState(data.state_values);

            // Only sync editBenches if we are NOT in interrupted mode (or first time entering)
            // But if we are in interrupted mode, we want to keep user edits.
            // However, if the backend adds new benches (e.g. from search), we want them.
            // Strategy: Only sync if status changed to interrupted, or if we are not editing.
            if (data.status === "interrupted" && status !== "interrupted") {
                 setEditBenches(data.state_values.benches || []);
            } else if (data.status === "interrupted" && Array.isArray(data.state_values.benches)) {
                const remoteBenches = data.state_values.benches;
                setEditBenches(prev => {
                    if (!Array.isArray(prev) || prev.length === 0) return prev;
                    const remoteMap = new Map(remoteBenches.map((b: any) => [b?.bench_name, b]));
                    return prev.map((b: any) => {
                        const rb = remoteMap.get(b?.bench_name);
                        if (!rb) return b;
                        const nextMeta = { ...(b?.meta || {}) };
                        const remoteMeta = rb?.meta || {};
                        if (remoteMeta && typeof remoteMeta === "object" && !Array.isArray(remoteMeta)) {
                            if (remoteMeta.download_error !== undefined) nextMeta.download_error = remoteMeta.download_error;
                        }
                        return {
                            ...b,
                            download_status: rb.download_status ?? b.download_status,
                            dataset_cache: rb.dataset_cache ?? b.dataset_cache,
                            eval_status: rb.eval_status ?? b.eval_status,
                            meta: nextMeta
                        };
                    });
                });
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
  }, [threadId, status, isResuming, t]);

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
              top_k: state.target_model.top_k ?? -1,
              repetition_penalty: state.target_model.repetition_penalty ?? 1.0,
              max_tokens: state.target_model.max_tokens ?? 2048,
              tensor_parallel_size: state.target_model.tensor_parallel_size ?? 1,
              max_model_len: state.target_model.max_model_len ?? 32768,
              gpu_memory_utilization: state.target_model.gpu_memory_utilization ?? 0.9,
              seed: state.target_model.seed ?? 0
          });
      } else if (state.target_model_name && !selectedModel && availableModels.length > 0) {
          const found = availableModels.find(m => m.name === state.target_model_name);
          if (found) setSelectedModel(found);
      }
  }, [state, availableModels.length, selectedModel]);

  // Auto-expand results
  useEffect(() => {
    if (!state?.benches) return;

    // Expand Metric Results when they are ready (success)
    const newExpandedMetricResults = [...expandedMetricResults];
    let changed = false;
    
    state.benches.forEach((b) => {
        const hasResult = !!b.meta?.eval_result;
        const isAlreadyExpanded = newExpandedMetricResults.includes(b.bench_name);
        
        if (hasResult && !isAlreadyExpanded) {
            newExpandedMetricResults.push(b.bench_name);
            changed = true;
        }
    });
    
    if (changed) {
        setExpandedMetricResults(newExpandedMetricResults);
    }

    // Expand Bench Cards if they are running or just finished
    const newExpandedResults = [...expandedResults];
    let resultsChanged = false;
    state.benches.forEach((b, idx) => {
         const isRunning = b.eval_status === "running";
         const isSuccess = b.eval_status === "success";
         const isExpanded = newExpandedResults.includes(idx);
         
         // Auto expand running or success benches if not already expanded
         if ((isRunning || isSuccess) && !isExpanded) {
             newExpandedResults.push(idx);
             resultsChanged = true;
         }
    });

    if (resultsChanged) {
        setExpandedResults(newExpandedResults);
    }

  }, [state?.benches]);

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
        target_model_path: targetModelPath,
        use_rag: useRAG,
        local_count: localCount,
        hf_count: hfCount,
        language: lang
      });
      setThreadId(res.data.thread_id);
      setStatus("running");
      
      setMessages(prev => [...prev, { id: Date.now().toString(), role: "ai", content: t({ zh: "我已启动评测流程，先为你解析需求。", en: "I've started the evaluation workflow. I'll analyze your query first." }), timestamp: Date.now() }]);

    } catch (e) {
      console.error(e);
      setMessages(prev => [...prev, { id: Date.now().toString(), role: "system", content: t({ zh: "启动流程失败，请检查服务连接。", en: "Failed to start workflow. Check connection." }), timestamp: Date.now() }]);
    }
  };

  const handleResume = async () => {
    if (!threadId) return;

    if (status === "interrupted" && currentNode?.includes("PreEvalReviewNode")) {
        const benchesToCheck = (editBenches.length ? editBenches : (state?.benches || [])) as any[];
        const missingEvalType = benchesToCheck
            .filter((b: any) => {
                const v = String(b?.bench_dataflow_eval_type || b?.eval_type || b?.meta?.bench_dataflow_eval_type || "").trim();
                return !v || v === "unknown";
            })
            .map((b: any) => b?.bench_name || "unknown");
        if (missingEvalType.length > 0) {
            setMessages(prev => [...prev, {
                id: Date.now().toString(),
                role: "system",
                content: t(
                    {
                        zh: `以下基准缺少 eval_type：${missingEvalType.join(", ")}。请打开对应数据集详情，在“评测类型”下拉中选择类型并保存后，再点击确认继续。eval_type 用于定义数据字段映射和评测方式。`,
                        en: `The following benches are missing eval_type: ${missingEvalType.join(", ")}. Open dataset details, select Evaluation Type from dropdown, save, then confirm to continue. eval_type defines field mapping and evaluation mode.`
                    }
                ),
                timestamp: Date.now()
            }]);
            return;
        }
    }
    
    setIsResuming(true);

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
              ? { 
                    ...nextModel, 
                    temperature: evalParams.temperature, 
                    top_p: evalParams.top_p, 
                    top_k: evalParams.top_k,
                    repetition_penalty: evalParams.repetition_penalty,
                    max_tokens: evalParams.max_tokens,
                    tensor_parallel_size: evalParams.tensor_parallel_size,
                    max_model_len: evalParams.max_model_len,
                    gpu_memory_utilization: evalParams.gpu_memory_utilization,
                    seed: evalParams.seed
                }
              : null;
          const shouldSendBenches = Boolean(
              status === "interrupted"
              && !currentNode?.includes("MetricReviewNode")
              && Array.isArray(editBenches)
              && editBenches.length > 0
          );
          payload.state_updates = {
              target_model_name: selectedModel?.name ?? state?.target_model_name,
              request: { language: lang }
          };
          if (shouldSendBenches) {
              payload.state_updates.benches = editBenches.length ? editBenches : (state?.benches || []);
          }
          if (modelForUpdate) {
              payload.state_updates.target_model = modelForUpdate;
          }
          if (currentNode?.includes("MetricReviewNode") && editMetricPlan) {
              payload.state_updates.metric_plan = editMetricPlan;
          }
      }

      await axios.post(`${apiBaseUrl}/api/workflow/resume/${threadId}`, payload);
      setStatus("running"); 
      setMessages(prev => [...prev, { id: Date.now().toString(), role: "ai", content: t({ zh: "已确认配置，继续执行评测流程。", en: "Configuration approved. Proceeding with evaluation..." }), timestamp: Date.now() }]);
      
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
          { id: "init", role: "ai", content: t({ zh: "已载入历史会话。", en: "Loaded past session." }), timestamp: Date.now() },
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
          { id: "init", role: "ai", content: t({ zh: "你好！请先描述你的评测目标，我们将自动开始流程。", en: "Hello! Describe your evaluation task to get started." }), timestamp: Date.now() }
      ]);
  };

  const handleSaveDraft = () => {
      const payload = {
          query,
          workMode,
          evalParams,
          manualModelPath,
          manualBenches,
          editBenches,
          useRAG,
          localCount,
          hfCount,
          savedAt: Date.now(),
      };
      localStorage.setItem(draftKey, JSON.stringify(payload));
      setMessages(prev => [
          ...prev,
          {
              id: Date.now().toString(),
              role: "system",
              content: t({ zh: "当前状态已保存到本地草稿。", en: "Current state has been saved to local draft." }),
              timestamp: Date.now(),
          },
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

      const inferredEvalTypeRaw =
          bench?.bench_dataflow_eval_type || bench?.eval_type || bench?.meta?.bench_dataflow_eval_type || "";
      const inferredEvalType = inferredEvalTypeRaw === "unknown" ? "" : String(inferredEvalTypeRaw || "");

      const newBench: Bench = {
          bench_name: bench.bench_name,
          eval_type: inferredEvalType,
          bench_dataflow_eval_type: inferredEvalType,
          meta: {
              ...bench.meta,
              tags: safeTaskTypes, // Store all task types as tags
              source: "gallery", // Flag to skip probing
              skip_probing: true,
              bench_dataflow_eval_type: inferredEvalType,
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

  const handleRetryDownload = async (params: { bench_name: string, config?: string, split?: string }) => {
      if (!threadId) return;
      const { bench_name, config, split } = params;

      const applyLocalPending = (b: any) => {
          if (b?.bench_name !== bench_name) return b;
          const nextMeta = { ...(b?.meta || {}) };
          const prevDl = nextMeta.download_config || {};
          nextMeta.download_config = { ...prevDl, ...(config ? { config } : {}), ...(split ? { split } : {}) };
          delete nextMeta.download_error;
          return { ...b, download_status: "pending", meta: nextMeta };
      };

      setEditBenches(prev => prev.map(applyLocalPending));
      setState(prev => {
          if (!prev) return prev;
          return { ...prev, benches: (prev.benches || []).map(applyLocalPending) };
      });

      await axios.post(`${apiBaseUrl}/api/workflow/redownload/${threadId}`, {
          bench_name,
          config,
          split
      });
  };

  const handleRerunExecution = async () => {
      if (!threadId) return;

      const benchesToSend = (status === "interrupted" ? editBenches : (state?.benches || [])) || [];
      const nextModel = selectedModel ?? state?.target_model ?? null;
      const modelForUpdate = nextModel
          ? { 
                ...nextModel, 
                temperature: evalParams.temperature, 
                top_p: evalParams.top_p, 
                top_k: evalParams.top_k,
                repetition_penalty: evalParams.repetition_penalty,
                max_tokens: evalParams.max_tokens,
                tensor_parallel_size: evalParams.tensor_parallel_size,
                max_model_len: evalParams.max_model_len,
                gpu_memory_utilization: evalParams.gpu_memory_utilization,
                seed: evalParams.seed
            }
          : null;
      const stateUpdates: any = {
          benches: benchesToSend,
          target_model_name: selectedModel?.name ?? state?.target_model_name,
      };
      if (modelForUpdate) stateUpdates.target_model = modelForUpdate;

      await axios.post(`${apiBaseUrl}/api/workflow/rerun_execution/${threadId}`, {
          state_updates: stateUpdates,
          goto_confirm: true
      });

      setStatus("running");
      setMessages(prev => [...prev, { id: Date.now().toString(), role: "ai", content: t({ zh: "已重新进入执行阶段，请确认配置后开始评测。", en: "Re-running execution. Please confirm configuration to start evaluation." }), timestamp: Date.now() }]);
  };

  const handleManualStart = async () => {
      const targetModelPath = manualModelPath || selectedModel?.path || "";
      if (!targetModelPath) {
          setMessages(prev => [...prev, { id: Date.now().toString(), role: "system", content: t({ zh: "手动模式：请先填写模型路径。", en: "Manual mode: please set a model path." }), timestamp: Date.now() }]);
          return;
      }
      if (!manualBenches.length) {
          setMessages(prev => [...prev, { id: Date.now().toString(), role: "system", content: t({ zh: "手动模式：请至少添加一个 Bench。", en: "Manual mode: please add at least one bench." }), timestamp: Date.now() }]);
          return;
      }

      const benchesPayload = manualBenches.map((b: any) => ({
          bench_name: b.bench_name,
          dataset_cache: b.dataset_cache,
          bench_dataflow_eval_type: b.bench_dataflow_eval_type || b.eval_type,
          meta: b.meta || {}
      }));

      const modelPayload: any = {
          model_name_or_path: targetModelPath,
          is_api: false,
          temperature: evalParams.temperature,
          top_p: evalParams.top_p,
          top_k: evalParams.top_k,
          repetition_penalty: evalParams.repetition_penalty,
          max_tokens: evalParams.max_tokens,
          tensor_parallel_size: evalParams.tensor_parallel_size,
          max_model_len: evalParams.max_model_len,
          gpu_memory_utilization: evalParams.gpu_memory_utilization,
          seed: evalParams.seed
      };

      const res = await axios.post(`${apiBaseUrl}/api/workflow/manual_start`, {
          user_query: query || "manual eval",
          target_model_name: selectedModel?.name || "manual",
          target_model: modelPayload,
          benches: benchesPayload
      });

      setThreadId(res.data.thread_id);
      setStatus("running");
      setMessages(prev => [...prev, { id: Date.now().toString(), role: "ai", content: t({ zh: "已启动手动评测，正在运行 DataFlowEval...", en: "Manual evaluation started. Running DataFlowEval..." }), timestamp: Date.now() }]);
  };

  const handleStopWorkflow = async () => {
      if (!threadId) return;
      try {
          await axios.post(`${apiBaseUrl}/api/workflow/stop/${threadId}`);
          setStatus("stopped");
          setMessages(prev => [
              ...prev,
              {
                  id: Date.now().toString(),
                  role: "system",
                  content: t({ zh: "已发送停止请求。", en: "Stop request has been sent." }),
                  timestamp: Date.now(),
              },
          ]);
      } catch (e) {
          setMessages(prev => [
              ...prev,
              {
                  id: Date.now().toString(),
                  role: "system",
                  content: t({ zh: "停止失败，请重试。", en: "Failed to stop. Please retry." }),
                  timestamp: Date.now(),
              },
          ]);
      }
  };

  const handleRunClick = async () => {
      if (status === "running") {
          await handleStopWorkflow();
          return;
      }
      if (workMode === "manual") {
          await handleManualStart();
          return;
      }
      if (status === "interrupted" && threadId) {
          await handleResume();
          return;
      }
      if (!query.trim()) {
          setMessages(prev => [
              ...prev,
              {
                  id: Date.now().toString(),
                  role: "system",
                  content: t({ zh: "请先在右侧聊天框输入评测需求，然后点击 Run。", en: "Please type your evaluation request in the right chat panel first, then click Run." }),
                  timestamp: Date.now(),
              },
          ]);
          return;
      }
      await handleStart(query.trim());
  };
  
  // Helper to determine block status
  const getBlockStatus = (block: 'search' | 'prep' | 'exec') => {
      if (status === 'idle') return 'idle';
      if (status === 'stopped') return 'interrupted';
      
      const nodes = currentNode ? [currentNode] : [];
      const isSearchActive = ["QueryUnderstandNode", "BenchSearchNode", "HumanReviewNode"].some(n => nodes.some(cn => cn.includes(n)));
      const isPrepActive = ["DatasetStructureNode", "BenchConfigRecommendNode", "BenchTaskInferNode", "DownloadNode"].some(n => nodes.some(cn => cn.includes(n)));
      const isExecActive = ["PreEvalReviewNode", "DataFlowEvalNode", "MetricRecommendNode", "MetricReviewNode", "ScoreCalcNode", "ReportGenNode"].some(n => nodes.some(cn => cn.includes(n)));

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
                       <span className="font-bold text-slate-800 truncate">{t({ zh: "历史记录", en: "History" })}</span>
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
                   title={t({ zh: "新建任务", en: "Start New Task" })}
               >
                   <Plus className={cn("w-4 h-4", !showHistory && "w-5 h-5")} />
                   {showHistory && <span>{t({ zh: "新任务", en: "New Task" })}</span>}
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
                                   <AlertTriangle className="w-3 h-3" /> {t({ zh: "确认删除？", en: "Confirm Delete?" })}
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
                                       {t({ zh: "取消", en: "Cancel" })}
                                   </Button>
                                   <Button 
                                       size="sm" 
                                       className="h-6 flex-1 text-[10px] p-0 bg-red-500 hover:bg-red-600 text-white" 
                                       onClick={(e) => handleDeleteHistory(e, safeThreadId)}
                                   >
                                       {t({ zh: "删除", en: "Delete" })}
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
                               title={t({ zh: "删除任务", en: "Delete Task" })}
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
                           )}>{item.status || t({ zh: "未知", en: "UNKNOWN" })}</span>
                           <span className="text-[10px] text-slate-400">{safeDate}</span>
                       </div>
                       <p className="text-xs text-slate-700 font-medium line-clamp-2 pr-4" title={item.user_query}>
                           {item.user_query || t({ zh: "未命名任务", en: "Untitled Task" })}
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
                    OneEval <span className="text-blue-600">{t({ zh: "工作台", en: "Studio" })}</span>
                </h2>
             </div>
             
             <div className="flex items-center gap-3">
                 <Button
                    variant="outline"
                    size="sm"
                    className="gap-2"
                    onClick={() => setLang(lang === "zh" ? "en" : "zh")}
                 >
                    {lang === "zh" ? "EN" : "中文"}
                 </Button>
                 <div className="flex items-center gap-2">
                     <span className="text-xs font-bold text-slate-500">{t({ zh: "智能", en: "Agent" })}</span>
                     <button
                         type="button"
                         onClick={() => setWorkMode(workMode === "agent" ? "manual" : "agent")}
                         className={cn(
                             "w-12 h-6 rounded-full relative transition-colors border",
                             workMode === "manual" ? "bg-emerald-500 border-emerald-600" : "bg-slate-200 border-slate-300"
                         )}
                     >
                         <span
                             className={cn(
                                 "absolute top-0.5 w-5 h-5 rounded-full bg-white shadow transition-all",
                                 workMode === "manual" ? "left-6" : "left-0.5"
                             )}
                         />
                     </button>
                     <span className="text-xs font-bold text-slate-500">{t({ zh: "手动", en: "Manual" })}</span>
                 </div>
                 <Button variant="outline" size="sm" className="gap-2">
                     <Database className="w-4 h-4" /> {t({ zh: "基准", en: "Benches" })}
                 </Button>
                 <Button variant="outline" size="sm" className="gap-2" onClick={() => setShowHistory(true)}>
                     <Layers className="w-4 h-4" /> {t({ zh: "任务", en: "Task" })}
                 </Button>
                 <Button variant="outline" size="sm" className="gap-2" onClick={handleSaveDraft}>
                     <Save className="w-4 h-4" /> {t({ zh: "保存", en: "Save" })}
                 </Button>
                 <div className="w-px h-6 bg-slate-200 mx-1" />
                 <Button 
                    size="sm" 
                    className={cn(
                        "gap-2 transition-all",
                        status === "running" ? "bg-red-500 hover:bg-red-600" : "bg-blue-600 hover:bg-blue-700"
                    )}
                   onClick={handleRunClick}
                 >
                     {status === "running"
                        ? <><X className="w-4 h-4" />{t({ zh: "停止", en: "Stop" })}</>
                        : <><Play className="w-4 h-4" />{t({ zh: "运行", en: "Run" })}</>
                      }
                 </Button>
             </div>
           </header>

           {/* Blocks Canvas */}
           <main className="flex-1 overflow-y-auto p-8 pb-32 scroll-smooth">
               {workMode === "manual" ? (
                   <div className="max-w-5xl mx-auto space-y-6">
                       <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6">
                           <div className="flex items-center justify-between gap-4">
                               <div>
                                   <div className="text-sm font-bold text-slate-900">Manual Evaluation</div>
                                   <div className="text-xs text-slate-500">Configure model and benches, then run DataFlowEval directly.</div>
                               </div>
                               <Button
                                   className="gap-2 bg-blue-600 hover:bg-blue-700"
                                   disabled={status === "running"}
                                   onClick={handleManualStart}
                               >
                                   <Play className="w-4 h-4" /> Run
                               </Button>
                           </div>
                       </div>

                       <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 space-y-4">
                           <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">Model</div>
                           <div className="grid grid-cols-12 gap-4">
                               <div className="col-span-6">
                                   <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Model Preset</label>
                                   <select
                                       value={(selectedModel?.name ?? state?.target_model_name ?? "") as any}
                                       onChange={(e) => {
                                           const found = availableModels.find((m: any) => m?.name === e.target.value);
                                           if (found) {
                                               setSelectedModel(found);
                                               if (found?.path) setManualModelPath(found.path);
                                           }
                                       }}
                                       disabled={status === "running"}
                                       className="w-full h-9 rounded-lg border border-slate-200 bg-white px-3 text-sm font-bold text-slate-900 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500/20 focus:border-blue-500 transition-all disabled:opacity-50 disabled:bg-slate-50/50"
                                   >
                                       {availableModels.map((m: any) => (
                                           <option key={m.name} value={m.name}>{m.name}</option>
                                       ))}
                                   </select>
                               </div>
                               <div className="col-span-6">
                                   <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Model Path</label>
                                   <Input
                                       value={manualModelPath}
                                       onChange={(e) => setManualModelPath(e.target.value)}
                                       disabled={status === "running"}
                                       className="h-9 bg-white border-slate-200 rounded-lg text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                   />
                               </div>
                           </div>
                       </div>

                       <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 space-y-4">
                           <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">Generation</div>
                           <div className="grid grid-cols-12 gap-x-4 gap-y-4">
                               <div className="col-span-3 min-w-0">
                                   <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Temperature</label>
                                   <Input
                                       type="number"
                                       step="0.1"
                                       min="0"
                                       max="2"
                                       value={evalParams.temperature}
                                       onChange={e => setEvalParams({ ...evalParams, temperature: parseFloat(e.target.value) || 0 })}
                                       disabled={status === "running"}
                                       className="h-9 bg-white border-slate-200 rounded-lg text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                   />
                               </div>
                               <div className="col-span-3 min-w-0">
                                   <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Top P</label>
                                   <Input
                                       type="number"
                                       step="0.05"
                                       min="0"
                                       max="1"
                                       value={evalParams.top_p}
                                       onChange={e => setEvalParams({ ...evalParams, top_p: parseFloat(e.target.value) || 0 })}
                                       disabled={status === "running"}
                                       className="h-9 bg-white border-slate-200 rounded-lg text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                   />
                               </div>
                               <div className="col-span-3 min-w-0">
                                   <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Top K</label>
                                   <Input
                                       type="number"
                                       step="1"
                                       value={evalParams.top_k}
                                       onChange={e => setEvalParams({ ...evalParams, top_k: parseInt(e.target.value) || 0 })}
                                       disabled={status === "running"}
                                       className="h-9 bg-white border-slate-200 rounded-lg text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                   />
                               </div>
                               <div className="col-span-3 min-w-0">
                                   <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Repetition</label>
                                   <Input
                                       type="number"
                                       step="0.05"
                                       min="0.5"
                                       max="2"
                                       value={evalParams.repetition_penalty}
                                       onChange={e => setEvalParams({ ...evalParams, repetition_penalty: parseFloat(e.target.value) || 1 })}
                                       disabled={status === "running"}
                                       className="h-9 bg-white border-slate-200 rounded-lg text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                   />
                               </div>

                               <div className="col-span-4 min-w-0">
                                   <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Max Tokens</label>
                                   <Input
                                       type="number"
                                       step="128"
                                       value={evalParams.max_tokens}
                                       onChange={e => setEvalParams({ ...evalParams, max_tokens: parseInt(e.target.value) || 0 })}
                                       disabled={status === "running"}
                                       className="h-9 bg-white border-slate-200 rounded-lg text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                   />
                               </div>
                               <div className="col-span-4 min-w-0">
                                   <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Tensor Parallel</label>
                                   <Input
                                       type="number"
                                       step="1"
                                       min="1"
                                       value={evalParams.tensor_parallel_size}
                                       onChange={e => setEvalParams({ ...evalParams, tensor_parallel_size: parseInt(e.target.value) || 1 })}
                                       disabled={status === "running"}
                                       className="h-9 bg-white border-slate-200 rounded-lg text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                   />
                               </div>
                               <div className="col-span-4 min-w-0">
                                   <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">GPU Mem Util</label>
                                   <Input
                                       type="number"
                                       step="0.05"
                                       min="0.1"
                                       max="1"
                                       value={evalParams.gpu_memory_utilization}
                                       onChange={e => setEvalParams({ ...evalParams, gpu_memory_utilization: parseFloat(e.target.value) || 0.9 })}
                                       disabled={status === "running"}
                                       className="h-9 bg-white border-slate-200 rounded-lg text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                   />
                               </div>
                               <div className="col-span-6 min-w-0">
                                   <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Max Model Len</label>
                                   <Input
                                       type="number"
                                       step="1024"
                                       value={evalParams.max_model_len}
                                       onChange={e => setEvalParams({ ...evalParams, max_model_len: parseInt(e.target.value) || 0 })}
                                       disabled={status === "running"}
                                       className="h-9 bg-white border-slate-200 rounded-lg text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                   />
                               </div>
                               <div className="col-span-6 min-w-0">
                                   <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Seed</label>
                                   <Input
                                       type="number"
                                       step="1"
                                       value={evalParams.seed}
                                       onChange={e => setEvalParams({ ...evalParams, seed: parseInt(e.target.value) || 0 })}
                                       disabled={status === "running"}
                                       className="h-9 bg-white border-slate-200 rounded-lg text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                   />
                               </div>
                           </div>
                       </div>

                       <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 space-y-4">
                           <div className="flex items-center justify-between">
                               <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">Benches</div>
                               <Button
                                   size="sm"
                                   variant="outline"
                                   className="gap-2"
                                   disabled={status === "running"}
                                   onClick={() => setManualBenches(prev => ([...prev, { bench_name: "", bench_dataflow_eval_type: "", dataset_cache: "", meta: {} } as any]))}
                               >
                                   <Plus className="w-4 h-4" /> Add Bench
                               </Button>
                           </div>

                           <div className="space-y-3">
                               {manualBenches.map((b: any, i: number) => (
                                   <div key={i} className="border border-slate-100 rounded-xl p-4 bg-slate-50/30">
                                       <div className="flex justify-between items-center mb-3">
                                           <div className="text-sm font-bold text-slate-700">Bench #{i + 1}</div>
                                           <Button
                                               size="sm"
                                               variant="ghost"
                                               className="h-7 px-2 text-slate-400 hover:text-red-500 hover:bg-red-50"
                                               disabled={status === "running"}
                                               onClick={() => setManualBenches(prev => prev.filter((_, idx) => idx !== i))}
                                           >
                                               <Trash2 className="w-4 h-4" />
                                           </Button>
                                       </div>

                                       <div className="grid grid-cols-12 gap-4">
                                           <div className="col-span-4">
                                               <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Bench Name</label>
                                               <Input
                                                   value={b.bench_name || ""}
                                                   onChange={(e) => setManualBenches(prev => prev.map((x, idx) => idx === i ? { ...x, bench_name: e.target.value } : x))}
                                                   disabled={status === "running"}
                                                   className="h-9 bg-white border-slate-200 rounded-lg text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                               />
                                           </div>
                                           <div className="col-span-4">
                                               <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Eval Type</label>
                                               <Input
                                                   value={b.bench_dataflow_eval_type || ""}
                                                   onChange={(e) => setManualBenches(prev => prev.map((x, idx) => idx === i ? { ...x, bench_dataflow_eval_type: e.target.value } : x))}
                                                   disabled={status === "running"}
                                                   className="h-9 bg-white border-slate-200 rounded-lg text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                               />
                                           </div>
                                           <div className="col-span-4">
                                               <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Dataset Cache</label>
                                               <Input
                                                   value={b.dataset_cache || ""}
                                                   onChange={(e) => setManualBenches(prev => prev.map((x, idx) => idx === i ? { ...x, dataset_cache: e.target.value } : x))}
                                                   disabled={status === "running"}
                                                   className="h-9 bg-white border-slate-200 rounded-lg text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                               />
                                           </div>

                                           <div className="col-span-12">
                                               <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Key Mapping (JSON)</label>
                                               <textarea
                                                   value={b.meta?.key_mapping_text ?? (b.meta?.key_mapping ? JSON.stringify(b.meta.key_mapping, null, 2) : "")}
                                                   onChange={(e) => {
                                                       const text = e.target.value;
                                                       setManualBenches(prev => prev.map((x, idx) => {
                                                           if (idx !== i) return x;
                                                           const nextMeta = { ...(x.meta || {}) };
                                                           nextMeta.key_mapping_text = text;
                                                           try {
                                                               nextMeta.key_mapping = JSON.parse(text);
                                                           } catch {}
                                                           return { ...x, meta: nextMeta };
                                                       }));
                                                   }}
                                                   disabled={status === "running"}
                                                   className="w-full min-h-[96px] p-3 bg-white rounded-lg border border-slate-200 text-xs font-mono text-slate-800 shadow-inner resize-y focus:outline-none focus:ring-2 focus:ring-blue-500/20 disabled:opacity-50 disabled:bg-slate-50/50"
                                               />
                                           </div>
                                       </div>
                                   </div>
                               ))}
                               {!manualBenches.length && (
                                   <div className="py-10 flex flex-col items-center justify-center text-slate-300 border-2 border-dashed border-slate-100 rounded-xl">
                                       <Database className="w-8 h-8 mb-2 opacity-50" />
                                       <span className="text-sm">Add benches to start manual evaluation</span>
                                   </div>
                               )}
                           </div>
                       </div>

                       <div className="bg-white rounded-2xl border border-slate-100 shadow-sm p-6 space-y-4">
                           <div className="flex items-center justify-between">
                               <div className="text-xs font-bold text-slate-500 uppercase tracking-wider">Progress</div>
                               {threadId && (
                                   <div className="text-[10px] text-slate-400 font-mono">thread: {threadId.slice(0, 8)}</div>
                               )}
                           </div>
                           {state?.benches?.length ? (
                               <div className="space-y-2">
                                   {state.benches.map((b: any, i: number) => (
                                       <div key={i} className="p-3 rounded-lg border border-slate-100 bg-slate-50/40 flex items-center justify-between">
                                           <div className="min-w-0">
                                               <div className="text-sm font-bold text-slate-700 truncate">{b.bench_name}</div>
                                               {b.eval_status === "running" && (
                                                   <div className="mt-2 h-1.5 w-56 bg-slate-100 rounded-full overflow-hidden">
                                                       <div className="h-full w-1/2 bg-blue-500/70 rounded-full animate-pulse" />
                                                   </div>
                                               )}
                                           </div>
                                           <span className={cn(
                                               "text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider shrink-0",
                                               b.eval_status === "success" ? "bg-emerald-50 text-emerald-700 border border-emerald-100" :
                                               b.eval_status === "running" ? "bg-blue-50 text-blue-700 border border-blue-100" :
                                               b.eval_status === "failed" ? "bg-red-50 text-red-700 border border-red-100" :
                                               "bg-slate-50 text-slate-500 border border-slate-100"
                                           )}>
                                               {b.eval_status || "pending"}
                                           </span>
                                       </div>
                                   ))}
                               </div>
                           ) : (
                               <div className="text-sm text-slate-400 italic">No running session yet.</div>
                           )}
                       </div>
                   </div>
               ) : (
               <div className="max-w-5xl mx-auto space-y-12">
                   
                   {/* Block 1: Discovery */}
                   <WorkflowBlock 
                        title={t({ zh: "发现阶段", en: "Discovery Phase" })} 
                        icon={Search}
                        activeNodeId={activeNode}
                        status={getBlockStatus('search') as any}
                        colorTheme="violet"
                        lang={lang}
                        nodes={[
                            { id: "QueryUnderstandNode", label: t({ zh: "理解", en: "Understand" }) },
                            { id: "BenchSearchNode", label: t({ zh: "检索", en: "Search" }) },
                            { id: "HumanReviewNode", label: t({ zh: "复核", en: "Review" }) }
                        ]}
                   >
                       <div className="space-y-8 relative">
                           {/* Node 1: Understand */}
                           <div className="pl-6 border-l-2 border-violet-100 relative">
                               <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-violet-50 border-2 border-violet-200" />
                              <h4 className="text-xs font-bold text-violet-600 uppercase tracking-wider mb-3">{t({ zh: "1. 理解需求", en: "1. Understand" })}</h4>
                               
                               <div className="space-y-3">
                                   <div>
                                      <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider pl-1">{t({ zh: "用户需求", en: "User Query" })}</label>
                                       <div className="p-3 bg-slate-50/50 rounded-lg border border-slate-100 text-sm text-slate-700 shadow-inner">
                                          {query || <span className="text-slate-400 italic">{t({ zh: "等待输入...", en: "Waiting for input..." })}</span>}
                                       </div>
                                   </div>
                                   {/* Domain (Placeholder/Mock if not in state) */}
                                   {(state as any)?.domain && (
                                       <div>
                                          <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider pl-1">{t({ zh: "识别领域", en: "Identified Domain" })}</label>
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

                           {/* Search Config: RAG toggle + Quota sliders */}
                           <div className="pl-6 border-l-2 border-violet-100 relative space-y-3">
                               <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-violet-50 border-2 border-violet-200" />
                               {/* RAG Toggle */}
                               <div className="flex items-center justify-between py-1">
                                   <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{t({ zh: "使用 RAG 推荐基准", en: "Use RAG for Benchmark Recommendation" })}</label>
                                   <button
                                       onClick={() => setUseRAG(!useRAG)}
                                       className={cn(
                                           "relative w-11 h-6 rounded-full transition-colors duration-200",
                                           useRAG ? "bg-violet-500" : "bg-slate-200"
                                       )}
                                   >
                                       <span className={cn(
                                           "absolute top-1 left-1 w-4 h-4 rounded-full bg-white shadow transition-transform duration-200",
                                           useRAG && "translate-x-5"
                                       )} />
                                   </button>
                               </div>
                               {/* Quota Controls */}
                               <div className="flex flex-col gap-1 py-1">
                                   <div className="flex items-center gap-4">
                                       <div className="flex items-center gap-2">
                                           <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider whitespace-nowrap">{t({ zh: "本地", en: "Local" })}</label>
                                           <div className="flex items-center gap-1">
                                               <button onClick={() => setLocalCount(Math.max(0, localCount - 1))} className="w-5 h-5 rounded bg-slate-100 hover:bg-slate-200 text-slate-500 text-xs flex items-center justify-center">-</button>
                                               <span className="w-5 text-center text-xs font-bold text-emerald-600">{localCount}</span>
                                               <button onClick={() => setLocalCount(Math.min(10, localCount + 1))} className="w-5 h-5 rounded bg-slate-100 hover:bg-slate-200 text-slate-500 text-xs flex items-center justify-center">+</button>
                                           </div>
                                       </div>
                                       <span className="text-slate-300 text-xs">+</span>
                                       <div className="flex items-center gap-2">
                                           <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider whitespace-nowrap">{t({ zh: "HF 搜索", en: "HF Search" })}</label>
                                           <div className="flex items-center gap-1">
                                               <button onClick={() => setHfCount(Math.max(0, hfCount - 1))} className="w-5 h-5 rounded bg-slate-100 hover:bg-slate-200 text-slate-500 text-xs flex items-center justify-center">-</button>
                                               <span className="w-5 text-center text-xs font-bold text-blue-600">{hfCount}</span>
                                               <button onClick={() => setHfCount(Math.min(10, hfCount + 1))} className="w-5 h-5 rounded bg-slate-100 hover:bg-slate-200 text-slate-500 text-xs flex items-center justify-center">+</button>
                                           </div>
                                       </div>
                                       <span className="text-[10px] text-slate-400">= {localCount + hfCount} {t({ zh: "个", en: "total" })}</span>
                                   </div>
                                   <p className="text-[10px] text-slate-400 leading-tight">
                                       {t({
                                           zh: "Local：从预置 Gallery 中检索，标记为 Gallery（已有完整评测配置）。HF Search：从 HuggingFace 在线搜索，需额外配置。由于本地Gallery检索结果与HF结果可能有重复，去重后实际数量可能略少于设定值。",
                                           en: "Local: from built-in Gallery with full eval config (marked Gallery). HF Search: online search from HuggingFace, needs extra config. For possible duplicates between local and HF results, actual count may be slightly less after dedup."
                                       })}
                                   </p>
                               </div>
                           </div>

                           {/* Node 2: Search */}
                           <div className="pl-6 border-l-2 border-violet-100 relative">
                               <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-violet-50 border-2 border-violet-200" />
                              <h4 className="text-xs font-bold text-violet-600 uppercase tracking-wider mb-3">{t({ zh: "2. 检索基准", en: "2. Search" })}</h4>
                               
                               {/* Editable Benches List */}
                               {state?.benches?.length ? (
                                   <div className="space-y-3">
                                       <div className="flex justify-between items-center px-1">
                                          <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{t({ zh: "目标基准", en: "Target Benchmarks" })}</label>
                                           {status === "interrupted" && (
                                               <div className="flex gap-2">
                                                   <Button size="sm" variant="outline" className="h-6 px-2 text-[10px] gap-1 bg-amber-50 border-amber-200 text-amber-700 hover:bg-amber-100" onClick={handleManualAdd}>
                                                        <Plus className="w-3 h-3" /> {t({ zh: "新增自定义", en: "Add Custom" })}
                                                   </Button>
                                                   <Button size="sm" variant="outline" className="h-6 px-2 text-[10px] gap-1 bg-blue-50 border-blue-200 text-blue-700 hover:bg-blue-100" onClick={() => setIsGalleryOpen(true)}>
                                                        <BookOpen className="w-3 h-3" /> {t({ zh: "从基准库添加", en: "From Gallery" })}
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
                                                   <div className="flex flex-col items-center gap-1 shrink-0">
                                                       <div className="w-8 h-8 rounded-lg bg-violet-100 text-violet-600 flex items-center justify-center text-xs font-bold">
                                                           {b.bench_name.substring(0, 2).toUpperCase()}
                                                       </div>
                                                       {b.meta?.source === 'hf_gallery' ? (
                                                           <span className="text-[8px] font-bold px-1 py-0.5 rounded bg-teal-100 text-teal-700 whitespace-nowrap" title={t({ zh: "HF搜索命中，且本地有完整配置，可直接使用", en: "Found via HF, full config in Gallery" })}>HF+Gallery</span>
                                                       ) : b.meta?.from_gallery ? (
                                                           <span className="text-[8px] font-bold px-1 py-0.5 rounded bg-emerald-100 text-emerald-700 whitespace-nowrap" title={t({ zh: "本地Gallery命中，有完整评测配置，可直接使用", en: "Local Gallery with full eval config" })}>Gallery</span>
                                                       ) : (
                                                           <span className="text-[8px] font-bold px-1 py-0.5 rounded bg-blue-100 text-blue-700 whitespace-nowrap" title={t({ zh: "来自HuggingFace搜索，需要额外配置", en: "From HF search, needs config" })}>HF</span>
                                                       )}
                                                   </div>
                                                   {status === "interrupted" ? (
                                                       <div className="flex flex-1 flex-col gap-1">
                                                           <div className="flex items-center gap-2">
                                                              <Input 
                                                                  placeholder={t({ zh: "输入 benchmark 名称...", en: "Enter benchmark name..." })}
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
                                                           {((lang === 'zh' ? (b.meta?.description_zh || b.meta?.description) : b.meta?.description) || b.meta?.desc) && (
                                                               <span className="text-xs text-slate-400 pl-1 line-clamp-2" title={(lang === 'zh' ? (b.meta?.description_zh || b.meta?.description) : b.meta?.description) || b.meta?.desc}>
                                                                   {(lang === 'zh' ? (b.meta?.description_zh || b.meta?.description) : b.meta?.description) || b.meta?.desc}
                                                               </span>
                                                           )}
                                                       </div>
                                                   ) : (
                                                       <div className="flex flex-col">
                                                           {/* Updated to show desc */}
                                                           <span className="font-mono font-medium text-sm text-slate-700">{b.bench_name}</span>
                                                           {((lang === 'zh' ? (b.meta?.description_zh || b.meta?.description) : b.meta?.description) || b.meta?.desc) && (
                                                               <span className="text-xs text-slate-400 max-w-md line-clamp-2" title={(lang === 'zh' ? (b.meta?.description_zh || b.meta?.description) : b.meta?.description) || b.meta?.desc}>
                                                                   {(lang === 'zh' ? (b.meta?.description_zh || b.meta?.description) : b.meta?.description) || b.meta?.desc}
                                                               </span>
                                                           )}
                                                       </div>
                                                   )}
                                               </div>
                                           ))}
                                       </div>
                                   </div>
                               ) : (
                                   <div className="text-sm text-slate-400 italic pl-1">{t({ zh: "暂无已选基准。", en: "No benchmarks selected yet." })}</div>
                               )}
                           </div>

                           {/* Node 3: Review */}
                           <div className="pl-6 border-l-2 border-transparent relative">
                               <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-violet-50 border-2 border-violet-200" />
                              <h4 className="text-xs font-bold text-violet-600 uppercase tracking-wider mb-3">{t({ zh: "3. 人工复核", en: "3. Review" })}</h4>
                               
                               <div className="space-y-2">
                                  <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider pl-1">{t({ zh: "补充说明", en: "User Notes" })}</label>
                                   <textarea 
                                       className="w-full p-3 bg-slate-50/50 rounded-lg border border-slate-100 text-sm text-slate-700 shadow-inner resize-none focus:outline-none focus:ring-1 focus:ring-violet-200"
                                       rows={2}
                                      placeholder={t({ zh: "补充本次评测的特殊要求或备注...", en: "Add any specific instructions or notes for this evaluation..." })}
                                   />
                               </div>
                           </div>
                       </div>
                   </WorkflowBlock>

                   {/* Block 2: Preparation */}
                   <WorkflowBlock 
                        title={t({ zh: "准备阶段", en: "Preparation Phase" })} 
                        icon={Database}
                        activeNodeId={activeNode}
                        status={getBlockStatus('prep') as any}
                        colorTheme="amber"
                        lang={lang}
                        nodes={[
                            { id: "DatasetStructureNode", label: t({ zh: "结构", en: "Structure" }) },
                            { id: "BenchConfigRecommendNode", label: t({ zh: "配置", en: "Config" }) },
                            { id: "BenchTaskInferNode", label: t({ zh: "推断", en: "Inference" }) },
                            { id: "DownloadNode", label: t({ zh: "下载", en: "Download" }) }
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
                                       lang={lang}
                                       onUpdate={(updated) => handleBenchUpdate(updated, i)}
                                       onRetryDownload={handleRetryDownload}
                                   />
                               </div>
                           ))}
                           {!(state?.benches?.length || editBenches.length) && (
                               <div className="col-span-2 py-8 flex flex-col items-center justify-center text-slate-300 border-2 border-dashed border-slate-100 rounded-xl">
                                   <Database className="w-8 h-8 mb-2 opacity-50" />
                                  <span className="text-sm">{t({ zh: "暂无已配置基准", en: "No benchmarks configured" })}</span>
                               </div>
                           )}
                       </div>
                       {(state?.benches?.length || editBenches.length) ? (
                           <div className="mt-4 text-xs text-slate-500 bg-amber-50/60 border border-amber-100 rounded-lg px-3 py-2">
                               {(() => {
                                   const source = (status === "interrupted" ? editBenches : state?.benches) || [];
                                   const total = source.length;
                                   const parsed = source.filter((b: any) => !!b?.meta?.structure?.ok || !!b?.meta?.structure_error).length;
                                   const downloaded = source.filter((b: any) => b?.download_status === "success").length;
                                   return t(
                                       { zh: `准备进度：结构解析 ${parsed}/${total}，数据下载 ${downloaded}/${total}`, en: `Preparation progress: structure ${parsed}/${total}, download ${downloaded}/${total}` }
                                   );
                               })()}
                           </div>
                       ) : null}
                   </WorkflowBlock>

                   {/* Block 3: Execution */}
                   <WorkflowBlock 
                        title={t({ zh: "执行阶段", en: "Execution Phase" })} 
                        icon={Play}
                        activeNodeId={activeNode}
                        status={getBlockStatus('exec') as any}
                        colorTheme="emerald"
                        lang={lang}
                        nodes={[
                            { id: "PreEvalReviewNode", label: t({ zh: "确认", en: "Confirm" }) },
                            { id: "DataFlowEvalNode", label: t({ zh: "评测", en: "Evaluation" }) },
                            { id: "MetricRecommendNode", label: t({ zh: "指标", en: "Metrics" }) },
                            { id: "MetricReviewNode", label: t({ zh: "复核", en: "Review" }) },
                            { id: "ScoreCalcNode", label: t({ zh: "计分", en: "Scoring" }) },
                            { id: "ReportGenNode", label: t({ zh: "报告", en: "Report" }) }
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
                                   <div className="flex items-center gap-3">
                                       <div className="flex items-center gap-2 text-emerald-800 font-bold text-sm">
                                           <Settings className="w-4 h-4" /> {t({ zh: "评测配置", en: "Evaluation Configuration" })}
                                       </div>
                                       {state?.benches?.length ? (
                                           <div className="text-[10px] font-bold text-slate-500 bg-white/70 border border-emerald-100 px-2 py-1 rounded">
                                               {(() => {
                                                   const total = state.benches.length;
                                                   const done = state.benches.filter((b: any) => b.eval_status === "success" || b.eval_status === "failed").length;
                                                   return t({ zh: `${done}/${total} 已完成`, en: `${done}/${total} done` });
                                               })()}
                                           </div>
                                       ) : null}
                                   </div>
                                   
                                   <div className="flex items-center gap-2">
                                      {(status === "completed" || status === "failed") && threadId && (
                                           <Button
                                               size="sm"
                                               variant="outline"
                                               className="h-7 text-xs gap-1"
                                               onClick={handleRerunExecution}
                                           >
                                               <RefreshCw className="w-3 h-3" /> {t({ zh: "重新执行", en: "Re-run Execution" })}
                                           </Button>
                                       )}

                                       {/* Status Indicator */}
                                       {status === "interrupted" && currentNode?.includes("PreEvalReviewNode") && (
                                           <span className="text-[10px] font-bold text-amber-600 bg-amber-100 px-2 py-1 rounded animate-pulse">
                                               {t({ zh: "等待确认", en: "Waiting for Confirmation" })}
                                           </span>
                                       )}
                                   </div>
                               </div>
                               {status === "interrupted" && currentNode?.includes("PreEvalReviewNode") && (() => {
                                   const source = (editBenches.length ? editBenches : (state?.benches || [])) as any[];
                                   const missing = source.filter((b: any) => !(b?.bench_dataflow_eval_type || b?.eval_type || b?.meta?.bench_dataflow_eval_type));
                                   const modelError = (state as any)?.error_msg;
                                   if (modelError && String(modelError).trim()) {
                                       return (
                                           <div className="mb-4 p-3 rounded-lg border border-red-200 bg-red-50 text-red-800 text-xs">
                                               {t(
                                                   {
                                                       zh: `模型加载失败：${modelError}。请先到 Settings 测试并修正模型路径，再返回此处点击确认继续。`,
                                                       en: `Model load failed: ${modelError}. Please test/fix model path in Settings, then return and confirm to continue.`
                                                   }
                                               )}
                                           </div>
                                       );
                                   }
                                   if (missing.length === 0) return null;
                                   return (
                                       <div className="mb-4 p-3 rounded-lg border border-amber-200 bg-amber-50 text-amber-800 text-xs">
                                           {t(
                                               {
                                                   zh: `有 ${missing.length} 个数据集缺少 eval_type。请进入对应数据集“详细信息”中的“评测类型”下拉完成配置后，再点击确认继续。`,
                                                   en: `${missing.length} datasets are missing eval_type. Open their details and set Evaluation Type from dropdown before confirming.`
                                               }
                                           )}
                                       </div>
                                   );
                               })()}
                               <div className="grid grid-cols-12 gap-x-4 gap-y-4">
                                   <div className="col-span-12 min-w-0">
                                       <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">{t({ zh: "目标模型", en: "Target Model" })}</label>
                                       {availableModels && availableModels.length > 0 ? (
                                           <select
                                               value={(selectedModel?.name ?? state?.target_model_name ?? "") as any}
                                               onChange={(e) => {
                                                   const found = availableModels.find((m: any) => m?.name === e.target.value);
                                                  if (found) {
                                                      setSelectedModel(found);
                                                      if (found?.path) setManualModelPath(found.path);
                                                  }
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
                                               {state?.target_model_name || t({ zh: "未选择模型", en: "No model selected" })}
                                           </div>
                                       )}
                                   </div>
                                   
                                   <div className="col-span-3 min-w-0">
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
                                   
                                   <div className="col-span-3 min-w-0">
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
                                   
                                   <div className="col-span-3 min-w-0">
                                       <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Top K</label>
                                       <Input 
                                           type="number" 
                                           step="1"
                                           value={evalParams.top_k} 
                                           onChange={e => setEvalParams({...evalParams, top_k: parseInt(e.target.value) || 0})}
                                           disabled={status === "running"}
                                           className="h-9 bg-white border-emerald-200 rounded-lg focus-visible:ring-emerald-500/20 focus-visible:border-emerald-500 text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                       />
                                   </div>

                                   <div className="col-span-3 min-w-0">
                                       <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Repetition</label>
                                       <Input 
                                           type="number" 
                                           step="0.05"
                                           min="0.5"
                                           max="2"
                                           value={evalParams.repetition_penalty} 
                                           onChange={e => setEvalParams({...evalParams, repetition_penalty: parseFloat(e.target.value) || 1})}
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

                                   <div className="col-span-4 min-w-0">
                                       <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Tensor Parallel</label>
                                       <Input 
                                           type="number" 
                                           step="1"
                                           min="1"
                                           value={evalParams.tensor_parallel_size} 
                                           onChange={e => setEvalParams({...evalParams, tensor_parallel_size: parseInt(e.target.value) || 1})}
                                           disabled={status === "running"}
                                           className="h-9 bg-white border-emerald-200 rounded-lg focus-visible:ring-emerald-500/20 focus-visible:border-emerald-500 text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                       />
                                   </div>

                                   <div className="col-span-4 min-w-0">
                                       <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">GPU Mem Util</label>
                                       <Input 
                                           type="number" 
                                           step="0.05"
                                           min="0.1"
                                           max="1"
                                           value={evalParams.gpu_memory_utilization} 
                                           onChange={e => setEvalParams({...evalParams, gpu_memory_utilization: parseFloat(e.target.value) || 0.9})}
                                           disabled={status === "running"}
                                           className="h-9 bg-white border-emerald-200 rounded-lg focus-visible:ring-emerald-500/20 focus-visible:border-emerald-500 text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                       />
                                   </div>

                                   <div className="col-span-6 min-w-0">
                                       <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Max Model Len</label>
                                       <Input 
                                           type="number" 
                                           step="1024"
                                           value={evalParams.max_model_len} 
                                           onChange={e => setEvalParams({...evalParams, max_model_len: parseInt(e.target.value) || 0})}
                                           disabled={status === "running"}
                                           className="h-9 bg-white border-emerald-200 rounded-lg focus-visible:ring-emerald-500/20 focus-visible:border-emerald-500 text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                       />
                                   </div>

                                   <div className="col-span-6 min-w-0">
                                       <label className="text-[10px] uppercase font-bold text-slate-400 mb-1.5 block px-1">Seed</label>
                                       <Input 
                                           type="number" 
                                           step="1"
                                           value={evalParams.seed} 
                                           onChange={e => setEvalParams({...evalParams, seed: parseInt(e.target.value) || 0})}
                                           disabled={status === "running"}
                                           className="h-9 bg-white border-emerald-200 rounded-lg focus-visible:ring-emerald-500/20 focus-visible:border-emerald-500 text-xs font-mono shadow-sm disabled:opacity-50 disabled:bg-slate-50/50"
                                       />
                                   </div>
                               </div>
                           </div>
                           {/* Metric Review Interrupt Block */}
                           {status === "interrupted" && currentNode?.includes("MetricReviewNode") && (editMetricPlan || (state?.metric_plan && Object.keys(state.metric_plan).length > 0)) && (
                               <div className="bg-white p-6 rounded-xl border-2 border-amber-200 shadow-lg shadow-amber-50 animate-in fade-in slide-in-from-bottom-4 duration-500">
                                   <div className="flex items-center justify-between mb-4">
                                       <div className="flex items-center gap-3">
                                           <div className="p-2 bg-amber-100 rounded-lg text-amber-600">
                                               <Bot className="w-5 h-5" />
                                           </div>
                                           <div>
                                               <h3 className="text-lg font-bold text-slate-800">{t({ zh: "复核指标方案", en: "Review Metrics Plan" })}</h3>
                                               <p className="text-sm text-slate-500">{t({ zh: "在计算得分前，为每个基准调整指标方案。", en: "Customize the metrics for each benchmark before calculation." })}</p>
                                           </div>
                                       </div>
                                       <Button 
                                           onClick={handleResume}
                                           className="bg-amber-500 hover:bg-amber-600 text-white gap-2 shadow-amber-200 shadow-lg"
                                       >
                                          <Check className="w-4 h-4" /> {t({ zh: "确认并计算得分", en: "Confirm & Calculate Scores" })}
                                       </Button>
                                   </div>

                                   <div className="grid grid-cols-1 gap-4">
                                       {Object.entries(editMetricPlan || state?.metric_plan || {}).map(([benchName, metrics]: [string, any[]]) => (
                                           <div key={benchName} className="border border-slate-200 rounded-lg p-4 bg-slate-50/50">
                                               <div className="flex items-center justify-between mb-3">
                                                   <span className="font-bold text-slate-700">{benchName}</span>
                                                   <div className="flex items-center gap-2 relative">
                                                       <span className="text-xs font-bold text-slate-400 bg-white px-2 py-1 rounded border border-slate-100">{t({ zh: `${metrics.length} 个指标`, en: `${metrics.length} metrics` })}</span>
                                                       
                                                       {/* Add Metric Button */}
                                                       <Button 
                                                           variant="ghost" 
                                                           size="icon" 
                                                           className="h-6 w-6 text-slate-400 hover:text-emerald-500 hover:bg-emerald-50"
                                                           onClick={() => {
                                                               setAddingMetricBench(addingMetricBench === benchName ? null : benchName);
                                                               setMetricSearch("");
                                                           }}
                                                       >
                                                           <Plus className="w-4 h-4" />
                                                       </Button>

                                                       {/* Add Metric Dropdown */}
                                                       {addingMetricBench === benchName && (
                                                           <div className="absolute top-8 right-0 z-50 w-64 bg-white border border-slate-200 rounded-lg shadow-xl p-2 animate-in fade-in zoom-in-95 duration-200">
                                                               <div className="relative mb-2">
                                                                   <Search className="absolute left-2 top-2.5 h-3.5 w-3.5 text-slate-400" />
                                                                   <Input 
                                                                       placeholder={t({ zh: "搜索指标...", en: "Search metrics..." })} 
                                                                       className="h-8 pl-8 text-xs"
                                                                       value={metricSearch}
                                                                       onChange={(e) => setMetricSearch(e.target.value)}
                                                                       autoFocus
                                                                   />
                                                               </div>
                                                               <div className="max-h-48 overflow-y-auto space-y-1">
                                                                   {metricRegistry
                                                                       .filter(m => 
                                                                           (m.name.toLowerCase().includes(metricSearch.toLowerCase()) || 
                                                                           m.desc.toLowerCase().includes(metricSearch.toLowerCase())) &&
                                                                           !metrics.some(existing => existing.name === m.name)
                                                                       )
                                                                       .map((m) => (
                                                                           <div 
                                                                               key={m.name} 
                                                                               className="flex flex-col p-2 hover:bg-slate-50 rounded cursor-pointer group"
                                                                               onClick={() => handleAddMetric(benchName, m)}
                                                                           >
                                                                               <span className="text-xs font-bold text-slate-700">{m.name}</span>
                                                                               <span className="text-[10px] text-slate-400 line-clamp-1">{m.desc}</span>
                                                                           </div>
                                                                       ))
                                                                   }
                                                                   {metricRegistry.filter(m => (m.name.toLowerCase().includes(metricSearch.toLowerCase()) || m.desc.toLowerCase().includes(metricSearch.toLowerCase())) && !metrics.some(existing => existing.name === m.name)).length === 0 && (
                                                                       <div className="p-2 text-center text-xs text-slate-400">
                                                                           {t({ zh: "无匹配指标", en: "No metrics found" })}
                                                                       </div>
                                                                   )}
                                                               </div>
                                                           </div>
                                                       )}
                                                   </div>
                                               </div>
                                               <div className="flex flex-wrap gap-3">
                                                   {metrics.map((m, idx) => {
                                                       const meta = metricRegistry.find(reg => reg.name === m.name || reg.aliases.includes(m.name));
                                                       return (
                                                           <div key={idx} className="bg-white p-3 rounded-lg border border-slate-200 text-sm flex flex-col gap-2 shadow-sm group relative w-full md:w-[calc(50%-0.75rem)] lg:w-[calc(33.33%-0.75rem)] transition-all hover:border-blue-200 hover:shadow-md">
                                                               <div className="flex justify-between items-start">
                                                                    <div className="flex items-center gap-2">
                                                                       <span className="font-bold text-slate-800">{m.name}</span>

                                                                    </div>
                                                                    {/* Delete Button */}
                                                                    <Button 
                                                                       variant="ghost" 
                                                                       size="icon" 
                                                                       className="h-6 w-6 text-slate-400 hover:text-red-500 hover:bg-red-50 -mt-1 -mr-1 opacity-0 group-hover:opacity-100 transition-opacity"
                                                                       onClick={() => {
                                                                           const newPlan = { ...(editMetricPlan || state?.metric_plan) };
                                                                           // Ensure array copy
                                                                           if (!editMetricPlan) {
                                                                               // Deep copy if first edit
                                                                               Object.keys(newPlan).forEach(k => {
                                                                                   newPlan[k] = [...newPlan[k]];
                                                                               });
                                                                           }
                                                                           
                                                                           newPlan[benchName] = newPlan[benchName].filter((_, i) => i !== idx);
                                                                           setEditMetricPlan(newPlan);
                                                                       }}
                                                                    >
                                                                        <X className="w-4 h-4" />
                                                                    </Button>
                                                               </div>
                                                               
                                                               {meta?.desc && (
                                                                   <div className="text-xs text-slate-500 line-clamp-2 leading-relaxed" title={meta.desc}>
                                                                       {meta.desc}
                                                                   </div>
                                                               )}
                                                               
                                                               {meta?.usage && (
                                                                   <div className="mt-auto pt-2 border-t border-slate-50 text-[10px] text-slate-400 flex items-start gap-1">
                                                                       <span className="font-bold uppercase text-slate-300 shrink-0">Usage:</span>
                                                                       <span className="line-clamp-1" title={meta.usage}>{meta.usage}</span>
                                                                   </div>
                                                               )}
                                                           </div>
                                                       );
                                                   })}
                                                   {metrics.length === 0 && (
                                                       <div className="w-full text-center py-4 text-slate-400 italic text-sm border-2 border-dashed border-slate-100 rounded-lg">
                                                           No metrics selected.
                                                       </div>
                                                   )}
                                               </div>
                                           </div>
                                       ))}
                                   </div>
                               </div>
                           )}
               
                           <div className="space-y-3">
                               {state?.benches?.map((b, i) => {
                                   const isExpanded = expandedResults.includes(i);
                                   const res = b.meta?.eval_result;
                                       const pickScore = (r: any) => {
                                           if (!r || typeof r !== 'object') return null;
                                           for (const k of ['score','exact_match','accuracy']) {
                                               const v = (r as any)[k];
                                               if (typeof v === 'number') return v;
                                           }
                                           return null;
                                       };
                                       const score = pickScore(res);
                                       const isRunningBench = b.eval_status === "running";
                                       const runningThisBench = isRunningBench && evalProgress?.bench_name === b.bench_name;
                                       const runningPercent = Math.max(0, Math.min(100, Number(evalProgress?.percent ?? 0)));
                                       const generated = Number(evalProgress?.generated ?? 0);
                                       const total = Number(evalProgress?.total ?? 0);
                                       const stage = String(evalProgress?.stage || "generator");
                                   
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
                                                       b.eval_status === "success" ? "bg-emerald-500" :
                                                       b.eval_status === "running" ? "bg-blue-500 animate-pulse" :
                                                       b.eval_status === "failed" ? "bg-red-500" :
                                                       "bg-slate-200"
                                                   )} />
                                                   <div>
                                                       <div className="flex items-center gap-2">
                                                           <div className="text-sm font-bold text-slate-700">{b.bench_name}</div>
                                                           <span className={cn(
                                                               "text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider",
                                                               b.eval_status === "success" ? "bg-emerald-50 text-emerald-700 border border-emerald-100" :
                                                               b.eval_status === "running" ? "bg-blue-50 text-blue-700 border border-blue-100" :
                                                               b.eval_status === "failed" ? "bg-red-50 text-red-700 border border-red-100" :
                                                               "bg-slate-50 text-slate-500 border border-slate-100"
                                                           )}>
                                                               {b.eval_status || "pending"}
                                                           </span>
                                                       </div>
                                                       <div className="text-[10px] text-slate-400 flex items-center gap-2">
                                                           {b.download_status === "success" && <span className="flex items-center gap-1"><Database className="w-3 h-3" /> {t({ zh: "已就绪", en: "Ready" })}</span>}
                                                           {b.eval_status === "success" && <span className="flex items-center gap-1 text-emerald-600"><Check className="w-3 h-3" /> {t({ zh: "已评测", en: "Evaluated" })}</span>}
                                                       </div>
                                                       {b.eval_status === "running" && (
                                                           <div className="mt-2 w-72">
                                                               <div className="h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                                                                   <div className={cn("h-full bg-blue-500/80 rounded-full transition-all", !runningThisBench ? "animate-pulse w-1/2" : "")} style={runningThisBench ? { width: `${runningPercent}%` } : undefined} />
                                                               </div>
                                                               <div className="mt-1 text-[10px] text-slate-500 font-mono">
                                                                   {runningThisBench
                                                                       ? t({ zh: `${stage === "evaluator" ? "评估中" : "生成中"} ${generated}/${total || "?"} (${runningPercent.toFixed(0)}%)`, en: `${stage === "evaluator" ? "Evaluating" : "Generating"} ${generated}/${total || "?"} (${runningPercent.toFixed(0)}%)` })
                                                                       : t({ zh: "准备评测中...", en: "Preparing evaluation..." })}
                                                               </div>
                                                           </div>
                                                       )}
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
                                                       <span className="text-xs text-slate-400 italic">{t({ zh: "等待结果...", en: "Waiting for results..." })}</span>
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
                                                           <div className="text-[10px] text-slate-400 uppercase font-bold mb-1">{t({ zh: "样本总数", en: "Total Samples" })}</div>
                                                           <div className="text-lg font-mono font-bold text-slate-700">
                                                               {b.meta?.eval_result?.total_samples || b.meta?.download_config?.count || b.meta?.structure?.count || "N/A"}
                                                           </div>
                                                       </div>
                                                       
                                                        <div className="p-3 bg-white rounded-lg border border-slate-100">
                                                            <div className="text-[10px] text-slate-400 uppercase font-bold mb-1">{t({ zh: "准确率（通用）", en: "Accuracy (General)" })}</div>
                                                            <div className="text-lg font-mono font-bold text-slate-700">
                                                                {/* 这里放你的数据，例如: b.meta?.eval_result?.accuracy || "0.00" */}
                                                                {b.meta?.eval_result?.accuracy || "N/A"} 
                                                            </div>
                                                        </div>

                                                       {/* Metric Cards Area */}
                                                       <div className="col-span-2 space-y-2">
                                                            <div 
                                                                className="text-[10px] text-slate-400 uppercase font-bold mb-1 cursor-pointer flex items-center justify-between hover:text-slate-600 transition-colors"
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    setExpandedMetricResults(prev => 
                                                                        prev.includes(b.bench_name) 
                                                                        ? prev.filter(name => name !== b.bench_name) 
                                                                        : [...prev, b.bench_name]
                                                                    );
                                                                }}
                                                            >
                                                                <span className="flex items-center gap-1"><Tag className="w-3 h-3" /> Metric Results</span>
                                                                {expandedMetricResults.includes(b.bench_name) ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                                                            </div>
                                                            {expandedMetricResults.includes(b.bench_name) && (
                                                                <div className="animate-in fade-in slide-in-from-top-1 duration-200">
                                                                    {res ? (
                                                                        <>
                                                                            <div className="h-64 w-full mb-4 bg-white p-4 rounded-lg border border-slate-100">
                                                                                <ResponsiveContainer width="100%" height="100%">
                                                                                    <BarChart
                                                                                        data={Object.entries(res)
                                                                                            .filter(([key]) => !["bench_name_or_prefix", "metric", "type", "valid_samples", "total_samples", "metric_summary_analyst", "case_study_analyst"].includes(key))
                                                                                            .map(([key, value]) => ({
                                                                                                name: key,
                                                                                                score: typeof value === 'number' ? value : parseFloat(String(value)) || 0
                                                                                            }))}
                                                                                        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
                                                                                    >
                                                                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                                                                        <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#64748b' }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} />
                                                                                        <YAxis tick={{ fontSize: 10, fill: '#64748b' }} axisLine={{ stroke: '#cbd5e1' }} tickLine={false} />
                                                                                        <Tooltip 
                                                                                            contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                                                                                            itemStyle={{ fontSize: '12px', color: '#334155' }}
                                                                                            labelStyle={{ fontSize: '12px', fontWeight: 'bold', color: '#0f172a', marginBottom: '4px' }}
                                                                                        />
                                                                                        <Bar dataKey="score" fill="#10b981" radius={[4, 4, 0, 0]} barSize={40} />
                                                                                    </BarChart>
                                                                                </ResponsiveContainer>
                                                                            </div>
                                                                            <div className="grid grid-cols-2 gap-3">
                                                                                {Object.entries(res)
                                                                                    .filter(([key]) => !["bench_name_or_prefix",
                                                                                         "metric", 
                                                                                         "type", 
                                                                                         "valid_samples", 
                                                                                         "total_samples", 
                                                                                         "metric_summary_analyst", 
                                                                                         "case_study_analyst"
                                                                                        ].includes(key))
                                                                                    .map(([key, value]) => {
                                                                                    const meta = metricRegistry.find(m => m.name === key || m.aliases.includes(key));
                                                                                    return (
                                                                                        <div key={key} className="bg-white rounded-lg border border-slate-200 p-3 shadow-sm hover:shadow-md transition-shadow relative overflow-hidden group">
                                                                                            <div className="absolute top-0 right-0 w-16 h-16 bg-gradient-to-bl from-slate-50 to-transparent opacity-50 rounded-bl-full pointer-events-none" />
                                                                                            
                                                                                            <div className="flex justify-between items-start mb-2 relative z-10">
                                                                                                <div className="flex flex-col">
                                                                                                    <span className="text-xs font-bold text-slate-700 flex items-center gap-2">
                                                                                                        {key}
                                                                                                    </span>
                                                                                                    {meta?.desc && (
                                                                                                        <span className="text-[10px] text-slate-400 mt-0.5 line-clamp-1" title={meta.desc}>
                                                                                                            {meta.desc}
                                                                                                        </span>
                                                                                                    )}
                                                                                                </div>
                                                                                                <span className="font-mono font-bold text-lg text-emerald-600">
                                                                                                    {(() => {
                                                                                                        const v = typeof value === 'string' ? parseFloat(value) : value;
                                                                                                        if (typeof v === 'number' && !isNaN(v)) {
                                                                                                            return Number.isInteger(v) ? v : v.toFixed(4);
                                                                                                        }
                                                                                                        return String(value);
                                                                                                    })()}
                                                                                                </span>
                                                                                            </div>
                                                                                            
                                                                                            {meta?.usage && (
                                                                                                <div className="mt-2 pt-2 border-t border-slate-50 text-[10px] text-slate-500 relative z-10">
                                                                                                    <span className="font-bold text-slate-300 uppercase mr-1">Usage:</span>
                                                                                                    {meta.usage}
                                                                                                </div>
                                                                                            )}
                                                                                        </div>
                                                                                    );
                                                                                })}
                                                                            </div>
                                                                        </>
                                                                    ) : (
                                                                        <div className="col-span-2 text-xs text-slate-400 italic p-2">{t({ zh: "暂无详细指标结果。", en: "No detailed metrics available yet." })}</div>
                                                                    )}
                                                                </div>
                                                            )}
                                                       </div>

                                                       {/* Recommended Metrics Section
                                                       {(state.metric_plan && state.metric_plan[b.bench_name]) && (
                                                           <div className="col-span-2 space-y-2 mt-2 pt-2 border-t border-slate-50">
                                                               <div className="text-[10px] text-violet-400 uppercase font-bold mb-1 flex items-center gap-2">
                                                                   <Bot className="w-3 h-3" /> Recommended Metrics Plan
                                                               </div>
                                                               <div className="flex flex-wrap gap-2">
                                                                   {state.metric_plan[b.bench_name].map((m: any, idx: number) => (
                                                                       <div key={idx} className="bg-white px-2 py-1 rounded border border-violet-100 text-xs flex items-center gap-2 shadow-sm" title={JSON.stringify(m.args || {})}>
                                                                           <span className="font-bold text-violet-700">{m.name}</span>
                                                                           {m.priority && (
                                                                               <span className={cn(
                                                                                   "text-[9px] px-1 rounded uppercase tracking-wider font-bold",
                                                                                   m.priority === "primary" ? "bg-violet-100 text-violet-600" : "bg-slate-100 text-slate-500"
                                                                               )}>
                                                                                   {m.priority}
                                                                               </span>
                                                                           )}
                                                                       </div>
                                                                   ))}
                                                               </div>
                                                           </div>
                                                       )} */}

                                                       {/* Metric Summary Text (New) */}
                                                       {b.meta?.metric_summary && (
                                                           <div className="col-span-2 space-y-2 mt-2 pt-2 border-t border-slate-50">
                                                               <div 
                                                                   className="text-[10px] text-blue-400 uppercase font-bold mb-1 cursor-pointer flex items-center justify-between hover:text-blue-600 transition-colors"
                                                                   onClick={(e) => {
                                                                       e.stopPropagation();
                                                                       setExpandedMetricSummaries(prev => 
                                                                           prev.includes(b.bench_name) 
                                                                           ? prev.filter(name => name !== b.bench_name) 
                                                                           : [...prev, b.bench_name]
                                                                       );
                                                                   }}
                                                               >
                                                                   <span className="flex items-center gap-1"><Bot className="w-3 h-3" /> Metric Summary</span>
                                                                   {expandedMetricSummaries.includes(b.bench_name) ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                                                               </div>
                                                               {expandedMetricSummaries.includes(b.bench_name) && (
                                                                   <div className="p-3 bg-slate-50 rounded-lg border border-slate-100 text-xs text-slate-600 font-sans animate-in fade-in slide-in-from-top-1 duration-200">
                                                                       <SimpleMarkdown content={b.meta.metric_summary} />
                                                                   </div>
                                                               )}
                                                           </div>
                                                       )}
                                                   </div>
                                                   {b.eval_status === "failed" && b.meta?.eval_error && (
                                                       <div className="mt-4 p-3 bg-red-50/50 rounded-lg border border-red-100 text-xs text-red-700 font-mono whitespace-pre-wrap">
                                                           {String(b.meta.eval_error)}
                                                       </div>
                                                   )}
                                               </div>
                                           )}
                                       </div>
                                   );
                               })}
                               
                               {!state?.benches?.length && (
                                   <div className="py-8 flex flex-col items-center justify-center text-slate-300 border-2 border-dashed border-slate-100 rounded-xl">
                                       <Play className="w-8 h-8 mb-2 opacity-50" />
                                      <span className="text-sm">{t({ zh: "准备执行", en: "Ready to execute" })}</span>
                                   </div>
                               )}
                               
                               {/* Summary Footer */}
                               {state?.benches?.length && state.benches.some(b => b.eval_status === "success") && (
                                   <div className="mt-6 p-4 bg-slate-800 text-white rounded-xl shadow-lg flex justify-between items-center">
                                       <div className="flex gap-6">
                                           <div>
                                              <div className="text-[10px] text-slate-400 uppercase font-bold">{t({ zh: "基准数量", en: "Benchmarks" })}</div>
                                               <div className="text-xl font-bold">{state.benches.length}</div>
                                           </div>
                                           <div>
                                              <div className="text-[10px] text-slate-400 uppercase font-bold">{t({ zh: "总样本数", en: "Total Samples" })}</div>
                                               <div className="text-xl font-bold">
                                                   {state.benches.reduce((acc, b) => acc + (parseInt(b.meta?.download_config?.count || b.meta?.structure?.count || 0)), 0)}
                                               </div>
                                           </div>
                                       </div>
                                       <div className="text-right">
                                          <div className="text-[10px] text-emerald-400 uppercase font-bold">{t({ zh: "总体状态", en: "Overall Status" })}</div>
                                          <div className="text-sm font-bold text-emerald-100">{t({ zh: "评测完成", en: "Evaluation Completed" })}</div>
                                       </div>
                                   </div>
                               )}
                           </div>
                       </div>
                   </WorkflowBlock>

              </div>
               )}
           </main>
           
           {/* Bottom Summary Panel */}
           <SummaryPanel 
                state={state} 
                sidebarWidth={showHistory ? 240 : 60} 
                chatWidth={chatWidth}
                lang={lang}
           />
       </div>

       {/* --- Right Sidebar (Chat) --- */}
       <div className="h-full z-40 shadow-2xl relative flex-shrink-0">
           <ChatPanel 
                messages={messages} 
                status={status}
                onSendMessage={handleStart}
                onConfirm={handleResume}
                onStop={handleStopWorkflow}
                isWaitingForInput={status !== "idle"}
                activeNodeId={activeNode} 
                isCollapsed={isChatCollapsed}
                onToggleCollapse={() => setIsChatCollapsed(!isChatCollapsed)}
                lang={lang}
                interruptToken={interruptToken}
           />
       </div>

       {/* Gallery Modal */}
       <GalleryModal 
            isOpen={isGalleryOpen} 
            onClose={() => setIsGalleryOpen(false)} 
            onSelect={handleGallerySelect}
            apiBaseUrl={apiBaseUrl}
            lang={lang}
       />

    </div>
  );
};

export default Eval;
