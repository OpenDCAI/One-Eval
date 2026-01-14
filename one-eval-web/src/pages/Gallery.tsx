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
} from "lucide-react";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

type BenchCategory = "Math" | "Reasoning" | "Knowledge" | "Safety" | "Coding" | "General";

type BenchMeta = {
  category: BenchCategory;
  tags: string[];
  description: string;
  sampleCount?: number;
};

type BenchItem = {
  id: string;
  name: string;
  meta: BenchMeta;
};

const DEFAULT_BENCHES: BenchItem[] = [
  {
    id: "gsm8k",
    name: "GSM8K",
    meta: {
      category: "Math",
      tags: ["Math", "Reasoning", "Arithmetic"],
      description: "Grade school math word problems, strong for multi-step reasoning.",
      sampleCount: 8792,
    },
  },
  {
    id: "math-500",
    name: "MATH-500",
    meta: {
      category: "Math",
      tags: ["Math", "Proof", "Algebra"],
      description: "A curated subset for harder mathematics and structured solutions.",
      sampleCount: 500,
    },
  },
  {
    id: "mmlu",
    name: "MMLU",
    meta: {
      category: "Knowledge",
      tags: ["Knowledge", "Multi-task", "QA"],
      description: "Broad academic and professional knowledge across many subjects.",
      sampleCount: 15908,
    },
  },
  {
    id: "truthfulqa",
    name: "TruthfulQA",
    meta: {
      category: "Safety",
      tags: ["Safety", "Hallucination", "Truthfulness"],
      description: "Measures whether a model avoids common misconceptions and hallucinations.",
      sampleCount: 817,
    },
  },
  {
    id: "humaneval",
    name: "HumanEval",
    meta: {
      category: "Coding",
      tags: ["Coding", "Python", "Unit Tests"],
      description: "Programming problems for code generation evaluation.",
      sampleCount: 164,
    },
  },
  {
    id: "hellaswag",
    name: "HellaSwag",
    meta: {
      category: "Reasoning",
      tags: ["Common Sense", "Reasoning", "Multiple Choice"],
      description: "Commonsense inference and continuation selection.",
      sampleCount: 10042,
    },
  },
];

const CATEGORIES: Array<{ id: BenchCategory | "All"; label: string }> = [
  { id: "All", label: "All" },
  { id: "Math", label: "Math" },
  { id: "Reasoning", label: "Reasoning" },
  { id: "Knowledge", label: "Knowledge" },
  { id: "Safety", label: "Safety" },
  { id: "Coding", label: "Coding" },
  { id: "General", label: "General" },
];

function getBenchIcon(category: BenchCategory) {
  switch (category) {
    case "Math":
      return { Icon: BookOpen, bg: "bg-emerald-50", fg: "text-emerald-600" };
    case "Reasoning":
      return { Icon: Brain, bg: "bg-indigo-50", fg: "text-indigo-600" };
    case "Knowledge":
      return { Icon: GraduationCap, bg: "bg-sky-50", fg: "text-sky-600" };
    case "Safety":
      return { Icon: ShieldCheck, bg: "bg-amber-50", fg: "text-amber-700" };
    case "Coding":
      return { Icon: Code2, bg: "bg-violet-50", fg: "text-violet-600" };
    default:
      return { Icon: Tag, bg: "bg-slate-50", fg: "text-slate-600" };
  }
}

function loadGalleryBenches(): BenchItem[] {
  try {
    const raw = localStorage.getItem("oneEval.gallery.benches");
    if (!raw) return DEFAULT_BENCHES;
    const parsed = JSON.parse(raw) as BenchItem[];
    if (!Array.isArray(parsed) || parsed.length === 0) return DEFAULT_BENCHES;
    return parsed;
  } catch {
    return DEFAULT_BENCHES;
  }
}

function saveGalleryBenches(items: BenchItem[]) {
  localStorage.setItem("oneEval.gallery.benches", JSON.stringify(items));
}

export const Gallery = () => {
  const navigate = useNavigate();
  const [benches, setBenches] = useState<BenchItem[]>([]);
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState<(typeof CATEGORIES)[number]["id"]>("All");
  const [activeBenchId, setActiveBenchId] = useState<string | null>(null);

  useEffect(() => {
    setBenches(loadGalleryBenches());
  }, []);

  const activeBench = useMemo(() => benches.find((b) => b.id === activeBenchId) ?? null, [benches, activeBenchId]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return benches
      .filter((b) => (category === "All" ? true : b.meta.category === category))
      .filter((b) => {
        if (!q) return true;
        const hay = `${b.name} ${b.meta.description} ${b.meta.tags.join(" ")} ${b.meta.category}`.toLowerCase();
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

  const handleReset = () => {
    setBenches(DEFAULT_BENCHES);
    saveGalleryBenches(DEFAULT_BENCHES);
  };

  return (
    <div className="p-12 max-w-7xl mx-auto space-y-8">
      <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4">
        <div className="space-y-1">
          <h1 className="text-4xl font-bold tracking-tight text-slate-900">Benchmark Gallery</h1>
          <p className="text-slate-600 text-lg">Search, filter, and configure your curated benchmarks.</p>
        </div>
        <div className="flex gap-3">
          <Button variant="outline" className="border-slate-200" onClick={handleReset}>
            Reset to Defaults
          </Button>
        </div>
      </div>

      <div className="flex flex-col gap-4">
        <div className="flex flex-col md:flex-row gap-3 md:items-center md:justify-between">
          <div className="relative w-full md:max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search benches, tags, categories..."
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
                    {c.label}
                  </button>
            ))}
          </div>
        </div>
      </div>

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
                        <div className="text-xs text-slate-500 mt-0.5">{bench.meta.category}</div>
                      </div>
                    </div>
                    {bench.meta.sampleCount != null && (
                      <div className="px-2 py-1 bg-slate-50 border border-slate-200 rounded text-xs font-mono text-slate-500">
                        {bench.meta.sampleCount.toLocaleString()} samples
                      </div>
                    )}
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
                  <CardDescription className="text-sm text-slate-600 line-clamp-3">{bench.meta.description}</CardDescription>
                </CardContent>

                <CardFooter className="pt-4 border-t border-slate-100 bg-slate-50/30 flex gap-2">
                  <Button
                    className="flex-1 text-white bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 shadow-sm shadow-blue-600/20"
                    onClick={() => handleUseBench(bench.id)}
                  >
                    Use
                  </Button>
                  <Button variant="outline" className="border-slate-200" onClick={() => setActiveBenchId(bench.id)}>
                    <SlidersHorizontal className="w-4 h-4 mr-2" />
                    Configure
                  </Button>
                </CardFooter>
              </Card>
            </motion.div>
          );
        })}
      </div>

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
                  <div className="text-xs text-slate-500 uppercase tracking-wider">Configure Bench</div>
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
                  <Label>Display Name</Label>
                  <Input
                    value={activeBench.name}
                    onChange={(e) => handleUpdateBench({ ...activeBench, name: e.target.value })}
                    className="border-slate-200"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Description</Label>
                  <textarea
                    value={activeBench.meta.description}
                    onChange={(e) =>
                      handleUpdateBench({ ...activeBench, meta: { ...activeBench.meta, description: e.target.value } })
                    }
                    className="w-full min-h-[120px] rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-300"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Category</Label>
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
                        {c.label}
                      </option>
                    ))}
                  </select>
                </div>

                <div className="space-y-2">
                  <Label>Tags (comma-separated)</Label>
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

                <div className="space-y-2">
                  <Label>Sample Count</Label>
                  <Input
                    type="number"
                    value={activeBench.meta.sampleCount ?? ""}
                    onChange={(e) =>
                      handleUpdateBench({
                        ...activeBench,
                        meta: {
                          ...activeBench.meta,
                          sampleCount: e.target.value ? Number(e.target.value) : undefined,
                        },
                      })
                    }
                    className="border-slate-200"
                  />
                </div>
              </div>

              <div className="mt-8 flex gap-2">
                <Button className="flex-1 bg-slate-900 text-white hover:bg-slate-800" onClick={() => handleUseBench(activeBench.id)}>
                  Use This Bench
                </Button>
                <Button variant="outline" className="border-slate-200" onClick={() => setActiveBenchId(null)}>
                  Close
                </Button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
