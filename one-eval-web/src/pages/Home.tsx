import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { ArrowRight, Activity, Zap, Layers, Github, Star } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useCallback, useEffect, useState } from "react";
import Particles from "react-tsparticles";
import { loadSlim } from "tsparticles-slim";
import type { Engine } from "tsparticles-engine";
import { useLang } from "@/lib/i18n";

export const Home = () => {
  const { lang, setLang, t } = useLang();
  const [stars, setStars] = useState({
    dataflow: null as number | null,
    oneEval: null as number | null,
  });
  const particlesInit = useCallback(async (engine: Engine) => {
    await loadSlim(engine);
  }, []);

  useEffect(() => {
    const loadStars = async () => {
      const fetchRepoStars = async (repo: string) => {
        try {
          const res = await fetch(`https://api.github.com/repos/${repo}`);
          if (!res.ok) return null;
          const data = await res.json();
          const stars = Number(data?.stargazers_count);
          return Number.isFinite(stars) ? stars : null;
        } catch {
          return null;
        }
      };
      try {
        const [dataflowStars, oneEvalStars] = await Promise.all([
          fetchRepoStars("OpenDCAI/DataFlow"),
          fetchRepoStars("OpenDCAI/One-Eval"),
        ]);
        setStars({
          dataflow: dataflowStars,
          oneEval: oneEvalStars,
        });
      } catch {}
    };
    loadStars();
  }, []);

  const formatStars = (count: number | null) => (typeof count === "number" ? Intl.NumberFormat("en-US").format(count) : "--");

  return (
    <div className="h-screen w-full bg-white flex flex-col font-['Inter'] overflow-hidden relative">
      
      {/* Particles Background */}
      <div className="absolute inset-0 z-0">
      <Particles
        id="tsparticles"
        init={particlesInit}
        className="w-full h-full"
        options={{
          background: {
            color: {
              value: "transparent",
            },
          },
          fpsLimit: 120,
          interactivity: {
            events: {
              onClick: {
                enable: true,
                mode: "push",
              },
              onHover: {
                enable: true,
                mode: "grab",
              },
              resize: true,
            },
            modes: {
              push: {
                quantity: 4,
              },
              grab: {
                distance: 140,
                links: {
                  opacity: 0.5,
                  color: "#2563eb", // blue-600
                },
              },
            },
          },
          particles: {
            color: {
              value: "#94a3b8", // slate-400
            },
            links: {
              color: "#cbd5e1", // slate-300
              distance: 150,
              enable: true,
              opacity: 0.3,
              width: 1,
            },
            move: {
              direction: "none",
              enable: true,
              outModes: {
                default: "bounce",
              },
              random: false,
              speed: 1,
              straight: false,
            },
            number: {
              density: {
                enable: true,
                area: 800,
              },
              value: 80,
            },
            opacity: {
              value: 0.5,
            },
            shape: {
              type: "circle",
            },
            size: {
              value: { min: 1, max: 3 },
            },
          },
          detectRetina: true,
        }}
      />
      </div>

      {/* Subtle Grid Background */}
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#f0f0f0_1px,transparent_1px),linear-gradient(to_bottom,#f0f0f0_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)] pointer-events-none" />

      {/* Navbar Placeholder */}
      <nav className="flex justify-between items-center px-8 py-6 z-10">
        <div className="text-2xl font-bold tracking-tight text-slate-900 flex items-center gap-3">
            <img src="/static/logo/logo.png" className="w-10 h-10 rounded-lg object-cover" alt="One-Eval" />
            <span>One-Eval</span>
        </div>
        <div className="flex items-center gap-2">
            <a
              href="https://github.com/OpenDCAI/DataFlow"
              target="_blank"
              rel="noreferrer"
              className="pl-3 pr-2 h-9 inline-flex items-center gap-2 rounded-full text-xs font-semibold text-slate-900 bg-white border border-slate-200 shadow-sm hover:shadow transition-shadow"
            >
              <Github className="w-4 h-4" />
              <span>DataFlow</span>
              <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-bold text-slate-700">
                <Star className="w-3 h-3 fill-slate-700 text-slate-700" />
                {formatStars(stars.dataflow)}
              </span>
            </a>
            <a
              href="https://github.com/OpenDCAI/One-Eval"
              target="_blank"
              rel="noreferrer"
              className="pl-3 pr-2 h-9 inline-flex items-center gap-2 rounded-full text-xs font-semibold text-slate-900 bg-white border border-slate-200 shadow-sm hover:shadow transition-shadow"
            >
              <Github className="w-4 h-4" />
              <span>One-Eval</span>
              <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 text-[11px] font-bold text-slate-700">
                <Star className="w-3 h-3 fill-slate-700 text-slate-700" />
                {formatStars(stars.oneEval)}
              </span>
            </a>
            <Button
              variant="outline"
              size="sm"
              className="rounded-full px-3 text-xs font-semibold border-slate-200"
              onClick={() => setLang(lang === "zh" ? "en" : "zh")}
            >
              {lang === "zh" ? "EN" : "中文"}
            </Button>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="flex-1 flex flex-col items-center justify-center text-center px-4 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="max-w-4xl space-y-8"
        >
          {/* Badge */}
          <div className="inline-flex items-center rounded-full border border-slate-200 bg-white px-3 py-1 text-sm text-slate-600 shadow-sm mb-4">
            <span className="flex h-2 w-2 rounded-full bg-blue-500 mr-2 animate-pulse"></span>
            {t({ zh: "v0.1.0 已发布", en: "v0.1.0 is now available" })}
          </div>

          <h1 className="text-5xl md:text-7xl font-bold tracking-tight text-slate-900 leading-[1.1]">
            One-Eval <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-600 to-violet-600">
              {t({
                zh: "一句话交给 Agent 完成评测",
                en: "evaluate in one via agents",
              })}
            </span>
          </h1>
          
          <p className="text-xl text-slate-600 max-w-2xl mx-auto leading-relaxed">
            {t({
              zh: "使用统一的图引擎编排复杂评测流程，从基准发现到细粒度指标分析，一站式完成。",
              en: "Orchestrate complex evaluation workflows with a unified, graph-based engine. From dataset discovery to granular metrics, all in one place.",
            })}
          </p>

          <div className="flex gap-4 justify-center pt-4">
            <Link to="/eval">
              <Button size="lg" className="h-12 px-8 text-base text-white bg-gradient-to-r from-blue-600 to-violet-600 hover:from-blue-500 hover:to-violet-500 shadow-lg shadow-blue-600/20">
                {t({ zh: "开始评测", en: "Start Evaluating" })}
                <ArrowRight className="ml-2 w-4 h-4" />
              </Button>
            </Link>
            <Link to="/gallery">
              <Button size="lg" variant="outline" className="h-12 px-8 text-base border-slate-200 hover:bg-slate-50">
                {t({ zh: "查看基准库", en: "View Gallery" })}
              </Button>
            </Link>
          </div>
        </motion.div>

        {/* Feature Highlights */}
        <motion.div 
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.4, duration: 0.8 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-20 max-w-5xl w-full px-4"
        >
            {[
                {
                  icon: Zap,
                  title: t({ zh: "快速配置", en: "Instant Setup" }),
                  desc: t({
                    zh: "连接模型后即可秒级启动评测任务。",
                    en: "Connect your model and start evaluating in seconds.",
                  }),
                },
                {
                  icon: Layers,
                  title: t({ zh: "图化引擎", en: "Graph Engine" }),
                  desc: t({
                    zh: "基于 LangGraph 构建复杂有状态流程。",
                    en: "Powered by LangGraph for complex, stateful workflows.",
                  }),
                },
                {
                  icon: Activity,
                  title: t({ zh: "深度指标", en: "Deep Metrics" }),
                  desc: t({
                    zh: "不仅看准确率，更提供细粒度分析洞察。",
                    en: "Get granular insights beyond just accuracy scores.",
                  }),
                }
            ].map((feature, i) => (
                <div key={i} className="flex flex-col items-center p-6 rounded-2xl bg-white border border-slate-100 shadow-sm hover:shadow-md transition-shadow">
                    <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center text-blue-600 mb-4">
                        <feature.icon className="w-6 h-6" />
                    </div>
                    <h3 className="font-semibold text-slate-900 mb-2">{feature.title}</h3>
                    <p className="text-sm text-slate-500">{feature.desc}</p>
                </div>
            ))}
        </motion.div>
      </main>
    </div>
  );
};
