import { motion } from "framer-motion";
import { cn } from "@/lib/utils";
import { SimpleMarkdown } from "@/components/ui/simple-markdown";
import { BarChart2, PieChart, Activity } from "lucide-react";
import type { Lang } from "@/lib/i18n";

// --- Types ---
interface RadarData {
  labels: string[];
  scores: number[];
}

interface DonutData {
  type: string;
  count: number;
  ratio: number;
}

interface HistogramData {
  bins: number[];
  correct: number[];
  incorrect: number[];
}

interface ReportData {
  version: string;
  generated_at: number;
  model: string;
  overall: {
    score: number;
    bench_summaries: any[];
  };
  macro: {
    radar: RadarData;
    sunburst: any; // We might skip sunburst or simplify it
    table: any[];
  };
  diagnostic: {
    error_distribution: DonutData[];
    length_histogram: HistogramData;
  };
  analyst: {
    metric_summary: Record<string, string>;
    case_study: Record<string, string>;
  };
  llm_summary: string;
}

// --- Components ---

// 1. Simple Radar Chart (SVG)
export const RadarChart = ({ data, size = 300 }: { data: RadarData; size?: number }) => {
  const { labels, scores } = data;
  const numPoints = labels.length;
  const radius = size / 2 - 40; // Padding
  const center = size / 2;
  
  if (numPoints < 3) return <div className="text-slate-400 italic">Not enough data for radar chart</div>;

  const getPoint = (index: number, value: number) => {
    const angle = (Math.PI * 2 * index) / numPoints - Math.PI / 2;
    const x = center + Math.cos(angle) * radius * value;
    const y = center + Math.sin(angle) * radius * value;
    return { x, y };
  };

  const levels = [0.2, 0.4, 0.6, 0.8, 1.0];
  const polyPoints = scores.map((s, i) => {
    const { x, y } = getPoint(i, s);
    return `${x},${y}`;
  }).join(" ");

  return (
    <div className="relative flex justify-center items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="overflow-visible">
        {/* Background Levels */}
        {levels.map((level, i) => (
          <polygon
            key={i}
            points={labels.map((_, j) => {
              const { x, y } = getPoint(j, level);
              return `${x},${y}`;
            }).join(" ")}
            fill="none"
            stroke="#e2e8f0"
            strokeWidth="1"
            strokeDasharray="4 4"
          />
        ))}

        {/* Axes */}
        {labels.map((_, i) => {
          const { x, y } = getPoint(i, 1.1);
          return (
            <line
              key={i}
              x1={center}
              y1={center}
              x2={x}
              y2={y}
              stroke="#e2e8f0"
              strokeWidth="1"
            />
          );
        })}

        {/* Data Polygon */}
        <motion.polygon
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 0.6, scale: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          points={polyPoints}
          fill="rgba(99, 102, 241, 0.2)" // Violet-500
          stroke="rgba(99, 102, 241, 0.8)"
          strokeWidth="2"
        />

        {/* Labels */}
        {labels.map((label, i) => {
          const { x, y } = getPoint(i, 1.25);
          return (
            <text
              key={i}
              x={x}
              y={y}
              textAnchor="middle"
              dominantBaseline="middle"
              className="text-[10px] font-bold fill-slate-500 uppercase tracking-wider"
              style={{ fontSize: '10px' }}
            >
              {label}
            </text>
          );
        })}
      </svg>
    </div>
  );
};

// 2. Simple Donut Chart (SVG)
export const DonutChart = ({ data, size = 200 }: { data: DonutData[]; size?: number }) => {
  const total = data.reduce((acc, d) => acc + d.count, 0);
  let accumulatedAngle = 0;
  const center = size / 2;
  const radius = size / 2 - 20;
  const thickness = 20;
  
  // Colors for error types
  const getColor = (type: string) => {
    if (type.includes("Correct")) return "#10b981"; // Emerald-500
    if (type.includes("Extraction")) return "#f59e0b"; // Amber-500
    if (type.includes("Refusal")) return "#64748b"; // Slate-500
    if (type.includes("Format")) return "#8b5cf6"; // Violet-500
    return "#ef4444"; // Red-500 (Incorrect/Logic)
  };

  if (total === 0) return <div className="text-slate-400 italic">No data</div>;

  return (
    <div className="flex items-center gap-8">
      <div className="relative" style={{ width: size, height: size }}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
            {data.map((d, i) => {
                const percentage = d.count / total;
                const angle = percentage * 360;
                const largeArcFlag = percentage > 0.5 ? 1 : 0;
                
                const startX = center + radius * Math.cos((accumulatedAngle - 90) * Math.PI / 180);
                const startY = center + radius * Math.sin((accumulatedAngle - 90) * Math.PI / 180);
                
                const endX = center + radius * Math.cos((accumulatedAngle + angle - 90) * Math.PI / 180);
                const endY = center + radius * Math.sin((accumulatedAngle + angle - 90) * Math.PI / 180);

                // For donut hole
                const innerRadius = radius - thickness;
                const startInnerX = center + innerRadius * Math.cos((accumulatedAngle - 90) * Math.PI / 180);
                const startInnerY = center + innerRadius * Math.sin((accumulatedAngle - 90) * Math.PI / 180);
                const endInnerX = center + innerRadius * Math.cos((accumulatedAngle + angle - 90) * Math.PI / 180);
                const endInnerY = center + innerRadius * Math.sin((accumulatedAngle + angle - 90) * Math.PI / 180);

                const path = `
                    M ${startX} ${startY}
                    A ${radius} ${radius} 0 ${largeArcFlag} 1 ${endX} ${endY}
                    L ${endInnerX} ${endInnerY}
                    A ${innerRadius} ${innerRadius} 0 ${largeArcFlag} 0 ${startInnerX} ${startInnerY}
                    Z
                `;
                
                const element = (
                    <motion.path
                        key={i}
                        d={path}
                        fill={getColor(d.type)}
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: i * 0.1 }}
                        className="hover:opacity-80 transition-opacity cursor-pointer"
                    />
                );
                
                accumulatedAngle += angle;
                return element;
            })}
            
            {/* Center Text */}
            <text x={center} y={center} textAnchor="middle" dominantBaseline="middle" className="fill-slate-700 font-bold text-xl">
                {total}
            </text>
            <text x={center} y={center + 20} textAnchor="middle" dominantBaseline="middle" className="fill-slate-400 text-xs uppercase font-bold tracking-wider">
                Total
            </text>
        </svg>
      </div>
      
      {/* Legend */}
      <div className="space-y-3">
          {data.map((d, i) => (
              <div key={i} className="flex items-center gap-3">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: getColor(d.type) }} />
                  <div>
                      <div className="text-xs font-bold text-slate-700">{d.type}</div>
                      <div className="text-[10px] text-slate-400">{d.count} samples ({Math.round(d.ratio * 100)}%)</div>
                  </div>
              </div>
          ))}
      </div>
    </div>
  );
};

// 3. Simple Bar Chart (HTML/CSS) for Benchmarks
export const BarChart = ({ data }: { data: { label: string; value: number; color?: string }[] }) => {
    return (
        <div className="space-y-3 w-full">
            {data.map((d, i) => (
                <div key={i} className="space-y-1">
                    <div className="flex justify-between text-xs">
                        <span className="font-bold text-slate-700 truncate max-w-[200px]" title={d.label}>{d.label}</span>
                        <span className="font-mono text-slate-500">{d.value.toFixed(2)}</span>
                    </div>
                    <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                        <motion.div 
                            initial={{ width: 0 }}
                            animate={{ width: `${d.value * 100}%` }}
                            transition={{ duration: 0.5, delay: i * 0.05 }}
                            className={cn("h-full rounded-full", d.color || "bg-blue-500")}
                        />
                    </div>
                </div>
            ))}
        </div>
    );
};

// 4. Histogram Chart (HTML/CSS) for Length Distribution
export const HistogramChart = ({ data, height = 200 }: { data: HistogramData; height?: number }) => {
    if (!data || !data.bins || data.bins.length === 0) return <div className="text-slate-400 italic">No data</div>;

    const maxVal = Math.max(...data.correct, ...data.incorrect, 1);
    
    return (
        <div className="w-full">
             <div className="flex items-end gap-1 w-full" style={{ height }}>
                {data.bins.map((bin, i) => {
                    const correctHeight = (data.correct[i] / maxVal) * 100;
                    const incorrectHeight = (data.incorrect[i] / maxVal) * 100;
                    
                    return (
                        <div key={i} className="flex-1 flex flex-col justify-end h-full gap-0.5 group relative">
                             {/* Tooltip */}
                             <div className="absolute bottom-full mb-2 left-1/2 -translate-x-1/2 bg-slate-800 text-white text-[10px] px-2 py-1 rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10 pointer-events-none">
                                 Len &lt; {bin}: {data.correct[i]} ✓ / {data.incorrect[i]} ✗
                             </div>

                             {/* Incorrect Bar (Top Stack) */}
                             {incorrectHeight > 0 && (
                                 <motion.div 
                                     initial={{ height: 0 }}
                                     animate={{ height: `${incorrectHeight}%` }}
                                     className="w-full bg-red-400 rounded-t-sm opacity-80 hover:opacity-100 transition-opacity"
                                 />
                             )}
                             
                             {/* Correct Bar (Bottom Stack) */}
                             {correctHeight > 0 && (
                                 <motion.div 
                                     initial={{ height: 0 }}
                                     animate={{ height: `${correctHeight}%` }}
                                     className="w-full bg-emerald-400 rounded-b-sm opacity-80 hover:opacity-100 transition-opacity"
                                 />
                             )}
                             
                             {/* X-Axis Label (Simplified) */}
                             {i % 2 === 0 && (
                                 <div className="absolute top-full mt-1 text-[9px] text-slate-400 text-center w-full">
                                     {bin}
                                 </div>
                             )}
                        </div>
                    );
                })}
             </div>
             <div className="mt-6 flex justify-center gap-4 text-[10px] text-slate-500 font-bold uppercase tracking-wider">
                 <div className="flex items-center gap-1"><div className="w-2 h-2 bg-emerald-400 rounded-sm"/> Correct</div>
                 <div className="flex items-center gap-1"><div className="w-2 h-2 bg-red-400 rounded-sm"/> Incorrect</div>
             </div>
        </div>
    );
};

// --- Main Report View Component ---
export const ReportView = ({ report, lang }: { report: ReportData, lang: Lang }) => {
    if (!report) return null;
    const tt = (zh: string, en: string) => (lang === "zh" ? zh : en);

    return (
        <div className="space-y-8 animate-in fade-in duration-500 pb-20">
            {/* Header Section (保持不变) */}
            <div className="bg-gradient-to-r from-slate-900 to-slate-800 rounded-2xl p-8 text-white shadow-xl relative overflow-hidden">
                <div className="absolute top-0 right-0 w-64 h-64 bg-white/5 rounded-full blur-3xl -mr-16 -mt-16 pointer-events-none" />
                
                <div className="relative z-10 flex justify-between items-start">
                    <div>
                        <div className="flex items-center gap-2 mb-2">
                            <Activity className="w-5 h-5 text-emerald-400" />
                            <span className="text-xs font-bold text-slate-200 uppercase tracking-wider">{tt("评测报告", "Evaluation Report")}</span>
                        </div>
                        <h1 className="text-3xl font-bold mb-2 text-white">{report.model}</h1>
                        <div className="text-white text-sm max-w-xl leading-relaxed">
                            <SimpleMarkdown content={report.llm_summary || tt("暂无摘要。", "No summary available.")} />
                        </div>
                    </div>
                    
                    <div className="text-right">
                        <div className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-1">{tt("综合得分", "Overall Score")}</div>
                        <div className="text-5xl font-black font-mono text-emerald-400 tracking-tight">
                            {report.overall.score.toFixed(4)}
                        </div>
                        <div className="text-xs text-slate-400 mt-2">
                            {tt("生成时间", "Generated at")} {new Date(report.generated_at * 1000).toLocaleDateString()}
                        </div>
                    </div>
                </div>
            </div>

            {/* Macro View Grid (保持不变) */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Radar Chart Card */}
                <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
                    <div className="flex items-center gap-2 mb-6">
                        <div className="p-2 bg-violet-100 text-violet-600 rounded-lg">
                            <Activity className="w-5 h-5" />
                        </div>
                        <h3 className="font-bold text-slate-800">{tt("能力雷达图", "Capabilities Radar")}</h3>
                    </div>
                    <div className="flex justify-center">
                        <RadarChart data={report.macro.radar} />
                    </div>
                </div>

                {/* Benchmark Performance Card */}
                <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
                    <div className="flex items-center gap-2 mb-6">
                        <div className="p-2 bg-blue-100 text-blue-600 rounded-lg">
                            <BarChart2 className="w-5 h-5" />
                        </div>
                        <h3 className="font-bold text-slate-800">{tt("基准得分", "Benchmark Scores")}</h3>
                    </div>
                    <div className="max-h-[300px] overflow-y-auto pr-2 scrollbar-thin">
                        <BarChart 
                            data={report.overall.bench_summaries.map(b => ({
                                label: b.bench,
                                value: b.primary_score,
                                color: b.primary_score >= 0.8 ? "bg-emerald-500" : b.primary_score >= 0.6 ? "bg-blue-500" : "bg-amber-500"
                            }))} 
                        />
                    </div>
                </div>
            </div>

            {/* Diagnostic Section */}
            <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm">
                <div className="flex items-center gap-2 mb-6">
                    <div className="p-2 bg-amber-100 text-amber-600 rounded-lg">
                        <PieChart className="w-5 h-5" />
                    </div>
                    <h3 className="font-bold text-slate-800">{tt("错误诊断", "Error Diagnostics")}</h3>
                </div>
                
                {/* 中间 Grid：左边甜甜圈，右边直方图 */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
                    {/* Left: Error Distribution */}
                    <div className="flex flex-col items-center">
                        <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4">{tt("错误分布", "Error Distribution")}</h4>
                        <DonutChart data={report.diagnostic.error_distribution} />
                    </div>
                    
                    {/* Right: Length Histogram */}
                    <div className="flex flex-col w-full">
                        <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4 flex items-center gap-2">
                             {tt("输出长度分布", "Output Length Distribution")}
                        </h4>
                        {report.diagnostic.length_histogram ? (
                            <div className="h-48 w-full">
                                <HistogramChart data={report.diagnostic.length_histogram} height={180} />
                            </div>
                        ) : (
                            <div className="h-48 flex items-center justify-center bg-slate-50 rounded-lg border border-dashed border-slate-200 text-slate-400 text-xs italic">
                                {tt("暂无长度分布数据", "No length data available")}
                            </div>
                        )}
                    </div>
                </div>

                {/* 底部 Footer：Analyst Insights */}
                <div className="mt-8 pt-8 border-t border-slate-100">
                     <h4 className="text-xs font-bold text-slate-500 uppercase tracking-wider mb-4">{tt("分析洞察", "Analyst Insights")}</h4>
                     {/* 使用 Grid 布局让文字卡片并排显示，避免单列太长 */}
                     <div className="space-y-4">
                         {Object.entries(report.analyst.metric_summary).slice(0, 6).map(([bench, text], i) => (
                             <div key={i} className="p-3 bg-slate-50 rounded-lg border border-slate-100 text-xs text-slate-600">
                                <strong className="block text-slate-800 mb-1">{bench}</strong>
                                <SimpleMarkdown content={text} />
                            </div>
                         ))}
                     </div>
                </div>
            </div>
        </div>
    );
};
