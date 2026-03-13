import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { AnimatePresence, motion } from "framer-motion";
import {
  BookOpen,
  Brain,
  Code2,
  GraduationCap,
  ShieldCheck,
  Search,
  SlidersHorizontal,
  Tag,
  X,
  ExternalLink,
  RefreshCw,
  Loader2,
  Plus,
  Bot,
  MessageSquare,
  FlaskConical,
  FileSearch,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { useLang } from "@/lib/i18n";

// ============================================================================
// Types - 匹配 bench_gallery.json 的数据结构
// ============================================================================

type BenchCategory = "Math" | "Reasoning" | "Knowledge & QA" | "Safety & Alignment" | "Coding" | "Agents & Tools" | "Instruction & Chat" | "Long Context & RAG" | "Domain-Specific";

// bench_gallery.json 中的 HF 元数据
type HfMeta = {
  bench_name: string;
  hf_repo: string;
  card_text: string;
  tags: string[];
  exists_on_hf: boolean;
};

// 数据集结构信息
type DatasetStructure = {
  repo_id: string;
  revision: string | null;
  subsets: Array<{
    subset: string;
    splits: Array<{
      name: string;
      num_examples: number | null;
    }>;
    features: Record<string, unknown> | null;
  }>;
  ok: boolean;
  error: string | null;
};

// 下载配置
type DownloadConfig = {
  config: string;
  split: string;
  reason: string;
};

// 字段映射
type KeyMapping = {
  input_question_key: string | null;
  input_target_key: string | null;
  input_context_key?: string | null;
};

// bench_gallery.json 中的 meta 字段
type BenchGalleryMeta = {
  bench_name: string;
  source: string;
  aliases: string[];
  category: BenchCategory;
  tags: string[];
  description: string;
  description_zh?: string;
  hf_meta: HfMeta;
  structure?: DatasetStructure;
  download_config?: DownloadConfig;
  key_mapping?: KeyMapping;
  key_mapping_reason?: string;
};

// bench_gallery.json 中的单个 bench 项
type BenchGalleryItem = {
  bench_name: string;
  bench_table_exist: boolean;
  bench_source_url: string;
  bench_dataflow_eval_type: string;
  bench_prompt_template: string | null;
  bench_keys: string[];
  meta: BenchGalleryMeta;
};

// 前端使用的简化类型（兼容旧代码）
type BenchItem = {
  id: string;
  name: string;
  meta: {
    category: BenchCategory;
    tags: string[];
    description: string;
    description_zh?: string;
    datasetUrl?: string;
    datasetKeys?: string[];
  };
  // 保留完整的原始数据
  _raw?: BenchGalleryItem;
};

// ============================================================================
// Constants
// ============================================================================

const CATEGORIES: Array<{ id: BenchCategory | "All" }> = [
  { id: "All" },
  { id: "Knowledge & QA" },
  { id: "Reasoning" },
  { id: "Math" },
  { id: "Coding" },
  { id: "Long Context & RAG" },
  { id: "Instruction & Chat" },
  { id: "Agents & Tools" },
  { id: "Safety & Alignment" },
  { id: "Domain-Specific" },
];

// ============================================================================
// Utility Functions
// ============================================================================

function getBenchIcon(category: BenchCategory) {
  switch (category) {
    case "Math":
      return { Icon: BookOpen, bg: "bg-emerald-50", fg: "text-emerald-600" };
    case "Reasoning":
      return { Icon: Brain, bg: "bg-indigo-50", fg: "text-indigo-600" };
    case "Knowledge & QA":
      return { Icon: GraduationCap, bg: "bg-sky-50", fg: "text-sky-600" };
    case "Safety & Alignment":
      return { Icon: ShieldCheck, bg: "bg-amber-50", fg: "text-amber-700" };
    case "Coding":
      return { Icon: Code2, bg: "bg-violet-50", fg: "text-violet-600" };
    case "Agents & Tools":
      return { Icon: Bot, bg: "bg-rose-50", fg: "text-rose-600" };
    case "Instruction & Chat":
      return { Icon: MessageSquare, bg: "bg-pink-50", fg: "text-pink-600" };
    case "Long Context & RAG":
      return { Icon: FileSearch, bg: "bg-cyan-50", fg: "text-cyan-600" };
    case "Domain-Specific":
      return { Icon: FlaskConical, bg: "bg-orange-50", fg: "text-orange-600" };
    default:
      return { Icon: Tag, bg: "bg-slate-50", fg: "text-slate-600" };
  }
}

/**
 * 将 bench_gallery.json 的数据转换为前端使用的 BenchItem 格式
 */
function transformBenchGalleryItem(item: BenchGalleryItem): BenchItem {
  const meta = item.meta || {};
  // 从 aliases 中获取显示名称（通常第二个是大写版本）
  const displayName = meta.aliases?.[1] || meta.aliases?.[0] || item.bench_name;

  return {
    id: item.bench_name,
    name: displayName,
    meta: {
      category: meta.category || "Knowledge & QA",
      tags: meta.tags || [],
      description: meta.description || "",
      description_zh: meta.description_zh || "",
      datasetUrl: item.bench_source_url,
      datasetKeys: item.bench_keys,
    },
    _raw: item,
  };
}

function getApiBaseUrl(): string {
  return localStorage.getItem("oneEval.apiBaseUrl") || "http://localhost:8000";
}

function loadGalleryBenches(): BenchItem[] {
  try {
    const raw = localStorage.getItem("oneEval.gallery.benches");
    if (!raw) return [];
    const parsed = JSON.parse(raw) as BenchItem[];
    if (!Array.isArray(parsed) || parsed.length === 0) return [];
    return parsed;
  } catch {
    return [];
  }
}

function saveGalleryBenches(items: BenchItem[]) {
  localStorage.setItem("oneEval.gallery.benches", JSON.stringify(items));
}

// ============================================================================
// Component
// ============================================================================

// Bench 类型选项（对应新分类）
const BENCH_TYPES = [
  "knowledge",
  "language & reasoning",
  "math",
  "coding",
  "information retrieval & RAG",
  "instruction-following",
  "conversation & chatbots",
  "agents & tools use",
  "safety",
  "bias & ethics",
  "domain-specific",
  "multilingual",
  "other",
];

export const Gallery = () => {
  const { lang } = useLang();
  const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);
  const categoryLabel = (id: BenchCategory | "All") => {
    const map: Record<BenchCategory | "All", { zh: string; en: string }> = {
      All: { zh: "全部", en: "All" },
      "Knowledge & QA": { zh: "知识问答", en: "Knowledge & QA" },
      Reasoning: { zh: "推理", en: "Reasoning" },
      Math: { zh: "数学", en: "Math" },
      Coding: { zh: "编程", en: "Coding" },
      "Long Context & RAG": { zh: "长上下文与 RAG", en: "Long Context & RAG" },
      "Instruction & Chat": { zh: "指令与对话", en: "Instruction & Chat" },
      "Agents & Tools": { zh: "Agent 与工具", en: "Agents & Tools" },
      "Safety & Alignment": { zh: "安全与对齐", en: "Safety & Alignment" },
      "Domain-Specific": { zh: "领域专项", en: "Domain-Specific" },
    };
    return tt(map[id].zh, map[id].en);
  };
  const benchTypeLabel = (value: string) => {
    const map: Record<string, { zh: string; en: string }> = {
      knowledge: { zh: "知识", en: "knowledge" },
      "language & reasoning": { zh: "语言与推理", en: "language & reasoning" },
      math: { zh: "数学", en: "math" },
      coding: { zh: "编程", en: "coding" },
      "information retrieval & RAG": { zh: "信息检索与 RAG", en: "information retrieval & RAG" },
      "instruction-following": { zh: "指令跟随", en: "instruction-following" },
      "conversation & chatbots": { zh: "对话与聊天机器人", en: "conversation & chatbots" },
      "agents & tools use": { zh: "Agent 与工具使用", en: "agents & tools use" },
      safety: { zh: "安全", en: "safety" },
      "bias & ethics": { zh: "偏见与伦理", en: "bias & ethics" },
      "domain-specific": { zh: "领域专项", en: "domain-specific" },
      multilingual: { zh: "多语言", en: "multilingual" },
      other: { zh: "其他", en: "other" },
    };
    const hit = map[value];
    return hit ? tt(hit.zh, hit.en) : value;
  };
  const navigate = useNavigate();
  const [benches, setBenches] = useState<BenchItem[]>([]);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<(typeof CATEGORIES)[number]["id"]>("All");
  const [activeBenchId, setActiveBenchId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Add Bench Modal 状态
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [addForm, setAddForm] = useState({
    bench_name: "",
    type: "knowledge",
    description: "",
    dataset_url: "",
  });

  // 从 API 获取 bench 数据
  const fetchBenches = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const apiBaseUrl = getApiBaseUrl();
      const response = await fetch(`${apiBaseUrl}/api/benches/gallery`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data: BenchGalleryItem[] = await response.json();
      const transformed = data.map(transformBenchGalleryItem);
      setBenches(transformed);
      saveGalleryBenches(transformed);
    } catch (err) {
      console.error("Failed to fetch benches:", err);
      setError(err instanceof Error ? err.message : tt("获取 Bench 失败", "Failed to fetch benches"));
      // 尝试从 localStorage 加载缓存数据
      const cached = loadGalleryBenches();
      if (cached.length > 0) {
        setBenches(cached);
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // 先尝试从缓存加载，然后从 API 刷新
    const cached = loadGalleryBenches();
    if (cached.length > 0) {
      setBenches(cached);
      setIsLoading(false);
    }
    fetchBenches();
  }, []);

  const activeBench = useMemo(() => benches.find((b) => b.id === activeBenchId) ?? null, [benches, activeBenchId]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return benches
      .filter((b) => (category === "All" ? true : b.meta.category === category))
      .filter((b) => {
        if (!q) return true;
        const hay = `${b.name} ${b.id} ${b.meta.description} ${b.meta.description_zh || ""} ${b.meta.tags.join(" ")} ${b.meta.category}`.toLowerCase();
        return hay.includes(q);
      });
  }, [benches, query, category]);

  const handleUseBench = (benchId: string) => {
    navigate("/eval", { state: { preSelectedBench: benchId } });
  };

  const handleUpdateBench = (updated: BenchItem) => {
    setBenches((prev) => {
      const next = prev.map((b) => (b.id === updated.id ? updated : b));
      saveGalleryBenches(next);
      return next;
    });
  };

  const handleRefresh = () => {
    fetchBenches();
  };

  const handleAddBench = async () => {
    if (!addForm.bench_name.trim() || !addForm.description.trim()) {
      return;
    }

    setIsSubmitting(true);
    try {
      const apiBaseUrl = getApiBaseUrl();
      const response = await fetch(`${apiBaseUrl}/api/benches/gallery`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          bench_name: addForm.bench_name.trim(),
          type: addForm.type,
          description: addForm.description.trim(),
          dataset_url: addForm.dataset_url.trim() || null,
        }),
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || tt("新增 Bench 失败", "Failed to add bench"));
      }

      // 成功后刷新列表并关闭弹窗
      await fetchBenches();
      setIsAddModalOpen(false);
      setAddForm({ bench_name: "", type: "knowledge", description: "", dataset_url: "" });
    } catch (err) {
      alert(err instanceof Error ? err.message : tt("新增 Bench 失败", "Failed to add bench"));
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="p-12 max-w-7xl mx-auto space-y-8">
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-4xl font-bold tracking-tight text-slate-900">{tt("基准库", "Benchmark Gallery")}</h1>
          <p className="text-slate-600 text-lg">{tt("搜索、筛选并配置你的精选基准。", "Search, filter, and configure your curated benchmarks.")}</p>
        </div>
        <div className="flex gap-3">
          <Button
            className="bg-gradient-to-r from-blue-600 to-violet-600 text-white hover:from-blue-500 hover:to-violet-500"
            onClick={() => setIsAddModalOpen(true)}
          >
            <Plus className="w-4 h-4 mr-2" />
            {tt("新增 Bench", "Add Bench")}
          </Button>
          <Button
            variant="outline"
            className="border-slate-200"
            onClick={handleRefresh}
            disabled={isLoading}
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <RefreshCw className="w-4 h-4 mr-2" />
            )}
            {tt("刷新", "Refresh")}
          </Button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}. {tt("已展示缓存数据。", "Showing cached data.")}
        </div>
      )}

      <div className="flex flex-col gap-4">
        <div className="flex flex-col md:flex-row gap-3 md:items-center md:justify-between">
          <div className="relative w-full md:max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={tt("搜索基准、标签、分类...", "Search benches, tags, categories...")}
              className="pl-9 bg-white border-slate-200"
            />
          </div>
          <div className="flex flex-wrap gap-2">
            {CATEGORIES.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => setCategory(c.id)}
                    className={cn(
                      "px-3 py-1.5 text-sm rounded-full border transition-colors",
                      c.id === category
                    ? "bg-gradient-to-r from-blue-600 to-violet-600 text-white border-transparent shadow-sm shadow-blue-600/20"
                    : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"
                    )}
                  >
                    {categoryLabel(c.id)}
                  </button>
            ))}
          </div>
        </div>
      </div>

      {isLoading && benches.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-slate-500">
          <Loader2 className="w-8 h-8 animate-spin mb-4" />
          <p>{tt("正在加载基准...", "Loading benchmarks...")}</p>
        </div>
      ) : benches.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-20 text-slate-500">
          <p>{tt("暂无可用基准。", "No benchmarks available.")}</p>
          <Button variant="outline" className="mt-4" onClick={handleRefresh}>
            {tt("重试", "Try Again")}
          </Button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {filtered.map((bench, idx) => {
            const { Icon, bg, fg } = getBenchIcon(bench.meta.category);
            return (
              <motion.div key={bench.id} initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: idx * 0.04 }}>
                <Card className="h-full flex flex-col border-slate-200 hover:shadow-lg transition-shadow duration-300">
                  <CardHeader>
                    <div className="flex justify-between items-start gap-4">
                      <div className="flex items-center gap-3">
                        <div className={cn("w-12 h-12 rounded-2xl flex items-center justify-center", bg)}>
                          <Icon className={cn("w-6 h-6", fg)} />
                        </div>
                        <div>
                          <CardTitle className="text-xl text-slate-900">{bench.name}</CardTitle>
                          <div className="text-xs text-slate-500 mt-0.5">{categoryLabel(bench.meta.category)}</div>
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-2 mt-4">
                      {bench.meta.tags.slice(0, 4).map((tag) => (
                        <span key={tag} className="text-xs px-2 py-0.5 rounded-full bg-slate-50 text-slate-600 border border-slate-200">
                          {tag}
                        </span>
                      ))}
                      {bench.meta.tags.length > 4 && (
                        <span className="text-xs px-2 py-0.5 rounded-full bg-slate-50 text-slate-500 border border-slate-200">
                          +{bench.meta.tags.length - 4}
                        </span>
                      )}
                    </div>
                  </CardHeader>

                  <CardContent className="flex-1">
                    <CardDescription className="text-sm text-slate-600 line-clamp-3">{lang === "zh" ? (bench.meta.description_zh || bench.meta.description) : bench.meta.description}</CardDescription>
                  </CardContent>

                  <CardFooter className="pt-4 border-t border-slate-100 bg-slate-50/30 flex gap-2">
                    <Button
                      className="flex-1 text-white bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 shadow-sm shadow-blue-600/20"
                      onClick={() => handleUseBench(bench.id)}
                    >
                      {tt("使用", "Use")}
                    </Button>
                    {bench.meta.datasetUrl && (
                      <Button
                        variant="outline"
                        className="border-slate-200"
                        onClick={() => window.open(bench.meta.datasetUrl, "_blank")}
                      >
                        <ExternalLink className="w-4 h-4" />
                      </Button>
                    )}
                    <Button variant="outline" className="border-slate-200" onClick={() => setActiveBenchId(bench.id)}>
                      <SlidersHorizontal className="w-4 h-4" />
                    </Button>
                  </CardFooter>
                </Card>
              </motion.div>
            );
          })}
        </div>
      )}

      <AnimatePresence>
        {activeBench && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50"
          >
            <div className="absolute inset-0 bg-black/20" onClick={() => setActiveBenchId(null)} />
            <motion.div
              initial={{ x: 420 }}
              animate={{ x: 0 }}
              exit={{ x: 420 }}
              transition={{ type: "spring", stiffness: 280, damping: 30 }}
              className="absolute right-0 top-0 bottom-0 w-full max-w-md bg-white border-l border-slate-200 shadow-2xl p-6 overflow-y-auto"
              role="dialog"
              aria-modal="true"
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-xs text-slate-500 uppercase tracking-wider">{tt("配置 Bench", "Configure Bench")}</div>
                  <div className="text-2xl font-bold text-slate-900 mt-1">{activeBench.name}</div>
                </div>
                <button
                  className="p-2 rounded-lg hover:bg-slate-100 text-slate-500"
                  onClick={() => setActiveBenchId(null)}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="mt-6 space-y-5">
                <div className="space-y-2">
                  <Label>{tt("显示名称", "Display Name")}</Label>
                  <Input
                    value={activeBench.name}
                    onChange={(e) => handleUpdateBench({ ...activeBench, name: e.target.value })}
                    className="border-slate-200"
                  />
                </div>

                <div className="space-y-2">
                  <Label>{tt("描述", "Description")}</Label>
                  <textarea
                    value={activeBench.meta.description}
                    onChange={(e) =>
                      handleUpdateBench({ ...activeBench, meta: { ...activeBench.meta, description: e.target.value } })
                    }
                    className="w-full min-h-[120px] rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                  />
                </div>

                <div className="space-y-2">
                  <Label>{tt("分类", "Category")}</Label>
                  <select
                    value={activeBench.meta.category}
                    onChange={(e) =>
                      handleUpdateBench({
                        ...activeBench,
                        meta: { ...activeBench.meta, category: e.target.value as BenchCategory },
                      })
                    }
                    className="w-full h-10 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                  >
                    {CATEGORIES.filter((c) => c.id !== "All").map((c) => (
                      <option key={c.id} value={c.id}>
                        {categoryLabel(c.id as BenchCategory)}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-2">
                  <Label>{tt("标签（逗号分隔）", "Tags (comma-separated)")}</Label>
                  <Input
                    value={activeBench.meta.tags.join(", ")}
                    onChange={(e) =>
                      handleUpdateBench({
                        ...activeBench,
                        meta: {
                          ...activeBench.meta,
                          tags: e.target.value
                            .split(",")
                            .map((t) => t.trim())
                            .filter(Boolean),
                        },
                      })
                    }
                    className="border-slate-200"
                  />
                </div>

                {activeBench.meta.datasetUrl && (
                  <div className="space-y-2">
                    <Label>{tt("数据集链接", "Dataset URL")}</Label>
                    <div className="flex gap-2">
                      <Input
                        value={activeBench.meta.datasetUrl}
                        readOnly
                        className="border-slate-200 bg-slate-50 text-slate-600 flex-1"
                      />
                      <Button
                        variant="outline"
                        className="border-slate-200 shrink-0"
                        onClick={() => window.open(activeBench.meta.datasetUrl, "_blank")}
                      >
                        <ExternalLink className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                )}

                {activeBench.meta.datasetKeys && activeBench.meta.datasetKeys.length > 0 && (
                  <div className="space-y-2">
                    <Label>{tt("数据字段 Keys", "Dataset Keys")}</Label>
                    <div className="flex flex-wrap gap-1.5 p-3 rounded-md border border-slate-200 bg-slate-50">
                      {activeBench.meta.datasetKeys.map((key) => (
                        <span key={key} className="text-xs px-2 py-1 rounded bg-white border border-slate-200 text-slate-600 font-mono">
                          {key}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {/* 显示额外的 bench_gallery.json 信息 */}
                {activeBench._raw && (
                  <>
                    {activeBench._raw.meta?.download_config && (
                      <div className="space-y-2">
                        <Label>{tt("下载配置", "Download Config")}</Label>
                        <div className="p-3 rounded-md border border-slate-200 bg-slate-50 text-xs font-mono space-y-1">
                          <div><span className="text-slate-500">config:</span> {activeBench._raw.meta.download_config.config}</div>
                          <div><span className="text-slate-500">split:</span> {activeBench._raw.meta.download_config.split}</div>
                        </div>
                      </div>
                    )}

                    {activeBench._raw.meta?.key_mapping && (
                      <div className="space-y-2">
                        <Label>{tt("字段映射", "Key Mapping")}</Label>
                        <div className="p-3 rounded-md border border-slate-200 bg-slate-50 text-xs font-mono space-y-1">
                          {activeBench._raw.meta.key_mapping.input_question_key && (
                            <div><span className="text-slate-500">question:</span> {activeBench._raw.meta.key_mapping.input_question_key}</div>
                          )}
                          {activeBench._raw.meta.key_mapping.input_target_key && (
                            <div><span className="text-slate-500">target:</span> {activeBench._raw.meta.key_mapping.input_target_key}</div>
                          )}
                          {activeBench._raw.meta.key_mapping.input_context_key && (
                            <div><span className="text-slate-500">context:</span> {activeBench._raw.meta.key_mapping.input_context_key}</div>
                          )}
                        </div>
                      </div>
                    )}

                    {activeBench._raw.bench_dataflow_eval_type && (
                      <div className="space-y-2">
                        <Label>{tt("评测类型", "Eval Type")}</Label>
                        <div className="px-3 py-2 rounded-md border border-slate-200 bg-slate-50 text-sm text-slate-600">
                          {activeBench._raw.bench_dataflow_eval_type}
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>

              <div className="mt-8 flex gap-2">
                <Button className="flex-1 bg-slate-900 text-white hover:bg-slate-800" onClick={() => handleUseBench(activeBench.id)}>
                  {tt("使用该 Bench", "Use This Bench")}
                </Button>
                <Button variant="outline" className="border-slate-200" onClick={() => setActiveBenchId(null)}>
                  {tt("关闭", "Close")}
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Add Bench Modal */}
      <AnimatePresence>
        {isAddModalOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center"
          >
            <div className="absolute inset-0 bg-black/20" onClick={() => setIsAddModalOpen(false)} />
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              className="relative bg-white rounded-2xl shadow-2xl p-6 w-full max-w-lg mx-4"
            >
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-slate-900">{tt("新增 Benchmark", "Add New Benchmark")}</h2>
                <button
                  className="p-2 rounded-lg hover:bg-slate-100 text-slate-500"
                  onClick={() => setIsAddModalOpen(false)}
                >
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-4">
                <div className="space-y-2">
                  <Label>{tt("Benchmark 名称 *", "Benchmark Name *")}</Label>
                  <Input
                    value={addForm.bench_name}
                    onChange={(e) => setAddForm({ ...addForm, bench_name: e.target.value })}
                    placeholder={tt("例如：org/dataset_name", "e.g., org/dataset_name")}
                    className="border-slate-200"
                  />
                  <p className="text-xs text-slate-500">{tt("请使用 HuggingFace 格式：org/dataset_name", "Use HuggingFace format: org/dataset_name")}</p>
                </div>

                <div className="space-y-2">
                  <Label>{tt("类型 *", "Type *")}</Label>
                  <select
                    value={addForm.type}
                    onChange={(e) => setAddForm({ ...addForm, type: e.target.value })}
                    className="w-full h-10 rounded-md border border-slate-200 bg-white px-3 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                  >
                    {BENCH_TYPES.map((t) => (
                      <option key={t} value={t}>{benchTypeLabel(t)}</option>
                    ))}
                  </select>
                </div>

                <div className="space-y-2">
                  <Label>{tt("描述 *", "Description *")}</Label>
                  <textarea
                    value={addForm.description}
                    onChange={(e) => setAddForm({ ...addForm, description: e.target.value })}
                    placeholder={tt("描述这个 benchmark 评测什么能力...", "Describe what this benchmark evaluates...")}
                    className="w-full min-h-[100px] rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                  />
                </div>

                <div className="space-y-2">
                  <Label>{tt("数据集链接（可选）", "Dataset URL (optional)")}</Label>
                  <Input
                    value={addForm.dataset_url}
                    onChange={(e) => setAddForm({ ...addForm, dataset_url: e.target.value })}
                    placeholder="https://huggingface.co/datasets/..."
                    className="border-slate-200"
                  />
                  <p className="text-xs text-slate-500">{tt("留空将根据 bench 名自动生成", "Leave empty to auto-generate from bench name")}</p>
                </div>
              </div>

              <div className="mt-6 flex gap-3">
                <Button
                  className="flex-1 bg-gradient-to-r from-blue-600 to-violet-600 text-white hover:from-blue-500 hover:to-violet-500"
                  onClick={handleAddBench}
                  disabled={isSubmitting || !addForm.bench_name.trim() || !addForm.description.trim()}
                >
                  {isSubmitting ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      {tt("新增中...", "Adding...")}
                    </>
                  ) : (
                    <>
                      <Plus className="w-4 h-4 mr-2" />
                      {tt("新增 Benchmark", "Add Benchmark")}
                    </>
                  )}
                </Button>
                <Button variant="outline" className="border-slate-200" onClick={() => setIsAddModalOpen(false)}>
                  {tt("取消", "Cancel")}
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
